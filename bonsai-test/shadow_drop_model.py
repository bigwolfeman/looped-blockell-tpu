"""Shadow-Drop: Train with shadow weights, then drop them.

Phase 1: Full STE + Adam (shadow weights bootstrap optimal ternary config)
Phase 2: DQT takes over (error buffer seeded from shadow residuals)

This is the "poor man's OBS" — use continuous training to FIND the
optimal ternary assignment, then switch to discrete-only optimization.

Memory timeline:
  Phase 1: ~10 bytes/param (shadow bf16 + Adam m1/m2)
  Phase 2: ~3 bytes/param (ternary int8 + error fp16 + scale fp16/128)

The key insight: shadow weights aren't needed for inference, and they
aren't needed for the MAJORITY of training either. They're only needed
for the initial "configuration discovery" phase.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm, ternary_quantize


class ShadowDropLinear(nn.Module):
    """Linear that transitions from STE+shadow to pure DQT mid-training."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 flip_rate: float = 0.05, error_decay: float = 0.999):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.flip_rate = flip_rate
        self.error_decay = error_decay

        total = out_features * in_features
        if total % group_size != 0:
            self.group_size = self._find_group_size(total, group_size)

        # Phase 1: shadow weight (trained by Adam)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # Phase 2 state (created at transition time)
        self.register_buffer("ternary", None)
        self.register_buffer("scales", None)
        self.register_buffer("error", None)
        self.register_buffer("scale_m1", None)
        self.register_buffer("scale_m2", None)
        self.scale_step = 0

        self._phase = 1  # 1 = STE+shadow, 2 = DQT

    @staticmethod
    def _find_group_size(total: int, target: int) -> int:
        for gs in range(target, 0, -1):
            if total % gs == 0:
                return gs
        return 1

    @property
    def phase(self):
        return self._phase

    def transition_to_dqt(self):
        """Drop shadow weights, switch to DQT mode.

        Seeds the error buffer with the shadow-ternary residual — this is
        the information that would be lost by dropping shadows. DQT then
        works to recover it through discrete steps.
        """
        with torch.no_grad():
            gs = self.group_size
            w = self.weight.data
            n_groups = w.numel() // gs

            # Quantize current shadow weight to ternary
            w_q, scales = ternary_quantize(w, gs)
            self.ternary = w_q.reshape(self.out_features, self.in_features).to(torch.int8)
            self.scales = scales

            # Seed error buffer with quantization residual
            # This is the "information" stored in the shadow weight beyond ternary
            scales_expanded = scales.unsqueeze(1).expand(n_groups, gs).reshape_as(w)
            reconstructed = self.ternary.float() * scales_expanded
            residual = w - reconstructed
            self.error = residual.clone()

            # Init scale optimizer
            self.scale_m1 = torch.zeros(n_groups, device=w.device)
            self.scale_m2 = torch.zeros(n_groups, device=w.device)
            self.scale_step = 0

        self._phase = 2
        # weight Parameter stays for gradient capture but is now overwritten each forward

    def forward(self, x: Tensor) -> Tensor:
        if self._phase == 1:
            # STE: quantize shadow weight, gradient passes through
            w_q, scales = ternary_quantize(self.weight, self.group_size)
            n_groups = self.weight.numel() // self.group_size
            scales_expanded = scales.unsqueeze(1).expand(n_groups, self.group_size)
            scales_expanded = scales_expanded.reshape(self.out_features, self.in_features)
            w_effective = w_q * scales_expanded
            # STE: use quantized for forward, gradient flows to shadow weight
            return F.linear(x, self.weight + (w_effective - self.weight).detach())
        else:
            # DQT: reconstruct from ternary
            with torch.no_grad():
                gs = self.group_size
                n_groups = self.ternary.numel() // gs
                s = self.scales.unsqueeze(1).expand(n_groups, gs)
                s = s.reshape(self.out_features, self.in_features)
                self.weight.data.copy_(self.ternary.float() * s)
            return F.linear(x, self.weight)

    def dqt_step(self, temperature: float = 1.0, scale_lr: float = 1e-3) -> dict:
        """DQT update — only active in phase 2."""
        if self._phase == 1 or self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        gs = self.group_size
        n_groups = self.ternary.numel() // gs

        effective_grad = grad + self.error
        scale_expanded = self.scales.unsqueeze(1).expand(n_groups, gs).reshape_as(self.ternary)
        normalized_grad = effective_grad / scale_expanded.float().clamp(min=1e-8)

        flip_prob = (normalized_grad.abs() * self.flip_rate / max(temperature, 0.01)).clamp(0, 0.5)
        direction = -normalized_grad.sign()

        do_flip = torch.bernoulli(flip_prob).bool()
        old_ternary = self.ternary.clone()

        step_dir = direction.sign().to(torch.int8)
        new_val = (self.ternary + step_dir).clamp(-1, 1)
        actually_changed = do_flip & (new_val != self.ternary)
        self.ternary[actually_changed] = new_val[actually_changed]

        was_zero = old_ternary == 0
        is_zero = self.ternary == 0
        n_sign = (actually_changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (actually_changed & (was_zero | is_zero)).sum().item()

        self.error[actually_changed] = 0.0
        self.error[~actually_changed] += grad[~actually_changed]
        self.error.mul_(self.error_decay)

        # Scale update
        self.scale_step += 1
        grad_scale = (grad * self.ternary.float()).reshape(-1, gs).sum(dim=1)
        b1, b2 = 0.9, 0.999
        self.scale_m1.lerp_(grad_scale, 1 - b1)
        self.scale_m2.lerp_(grad_scale.square(), 1 - b2)
        m1_hat = self.scale_m1 / (1 - b1 ** self.scale_step)
        m2_hat = self.scale_m2 / (1 - b2 ** self.scale_step)
        self.scales.addcdiv_(m1_hat, m2_hat.sqrt().add_(1e-8), value=-scale_lr)
        self.scales.clamp_(min=1e-6)

        self.weight.grad = None
        return {"n_flips": n_sign + n_structural, "n_sign": n_sign, "n_structural": n_structural}


class ShadowDropEmbedding(nn.Module):
    """Embedding that transitions from STE+shadow to pure DQT."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 flip_rate: float = 0.05, error_decay: float = 0.999):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.flip_rate = flip_rate
        self.error_decay = error_decay

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = ShadowDropLinear._find_group_size(total, group_size)

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

        self.register_buffer("ternary", None)
        self.register_buffer("scales", None)
        self.register_buffer("error", None)
        self.register_buffer("scale_m1", None)
        self.register_buffer("scale_m2", None)
        self.scale_step = 0
        self._phase = 1

    @property
    def phase(self):
        return self._phase

    def transition_to_dqt(self):
        with torch.no_grad():
            gs = self.group_size
            w = self.weight.data
            n_groups = w.numel() // gs

            w_q, scales = ternary_quantize(w, gs)
            self.ternary = w_q.reshape(self.num_embeddings, self.embedding_dim).to(torch.int8)
            self.scales = scales

            scales_expanded = scales.unsqueeze(1).expand(n_groups, gs).reshape_as(w)
            reconstructed = self.ternary.float() * scales_expanded
            self.error = (w - reconstructed).clone()

            self.scale_m1 = torch.zeros(n_groups, device=w.device)
            self.scale_m2 = torch.zeros(n_groups, device=w.device)
            self.scale_step = 0
        self._phase = 2

    def forward(self, input_ids: Tensor) -> Tensor:
        if self._phase == 1:
            w_q, scales = ternary_quantize(self.weight, self.group_size)
            n_groups = self.weight.numel() // self.group_size
            scales_expanded = scales.unsqueeze(1).expand(n_groups, self.group_size)
            scales_expanded = scales_expanded.reshape(self.num_embeddings, self.embedding_dim)
            w_effective = w_q * scales_expanded
            w_ste = self.weight + (w_effective - self.weight).detach()
            return F.embedding(input_ids, w_ste)
        else:
            with torch.no_grad():
                gs = self.group_size
                n_groups = self.ternary.numel() // gs
                s = self.scales.unsqueeze(1).expand(n_groups, gs)
                s = s.reshape(self.num_embeddings, self.embedding_dim)
                self.weight.data.copy_(self.ternary.float() * s)
            return F.embedding(input_ids, self.weight)

    def dqt_step(self, temperature: float = 1.0, scale_lr: float = 1e-3) -> dict:
        if self._phase == 1 or self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        gs = self.group_size
        n_groups = self.ternary.numel() // gs

        effective_grad = grad + self.error
        scale_expanded = self.scales.unsqueeze(1).expand(n_groups, gs).reshape_as(self.ternary)
        normalized_grad = effective_grad / scale_expanded.float().clamp(min=1e-8)

        flip_prob = (normalized_grad.abs() * self.flip_rate / max(temperature, 0.01)).clamp(0, 0.5)
        direction = -normalized_grad.sign()

        do_flip = torch.bernoulli(flip_prob).bool()
        old_ternary = self.ternary.clone()

        step_dir = direction.sign().to(torch.int8)
        new_val = (self.ternary + step_dir).clamp(-1, 1)
        actually_changed = do_flip & (new_val != self.ternary)
        self.ternary[actually_changed] = new_val[actually_changed]

        was_zero = old_ternary == 0
        is_zero = self.ternary == 0
        n_sign = (actually_changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (actually_changed & (was_zero | is_zero)).sum().item()

        self.error[actually_changed] = 0.0
        self.error[~actually_changed] += grad[~actually_changed]
        self.error.mul_(self.error_decay)

        self.scale_step += 1
        grad_scale = (grad * self.ternary.float()).reshape(-1, gs).sum(dim=1)
        b1, b2 = 0.9, 0.999
        self.scale_m1.lerp_(grad_scale, 1 - b1)
        self.scale_m2.lerp_(grad_scale.square(), 1 - b2)
        m1_hat = self.scale_m1 / (1 - b1 ** self.scale_step)
        m2_hat = self.scale_m2 / (1 - b2 ** self.scale_step)
        self.scales.addcdiv_(m1_hat, m2_hat.sqrt().add_(1e-8), value=-scale_lr)
        self.scales.clamp_(min=1e-6)

        self.weight.grad = None
        return {"n_flips": n_sign + n_structural, "n_sign": n_sign, "n_structural": n_structural}


class ShadowDropAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        gs = cfg.group_size
        fr = cfg.flip_rate
        ed = cfg.error_decay

        self.W_q = ShadowDropLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)
        self.W_k = ShadowDropLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)
        self.W_v = ShadowDropLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)
        self.W_o = ShadowDropLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)

        self.alpha = nn.Parameter(torch.full((cfg.n_heads, 1, 1), 0.1))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.W_q(x).reshape(B, S, H, Dh).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, H, Dh).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        x_heads = x.reshape(B, S, H, Dh).transpose(1, 2)
        out = out + self.alpha * x_heads
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_o(out)


class ShadowDropMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gs = cfg.group_size
        fr = cfg.flip_rate
        ed = cfg.error_decay
        self.w_gate = ShadowDropLinear(cfg.d_model, cfg.d_ff, group_size=gs, flip_rate=fr, error_decay=ed)
        self.w_up = ShadowDropLinear(cfg.d_model, cfg.d_ff, group_size=gs, flip_rate=fr, error_decay=ed)
        self.w_down = ShadowDropLinear(cfg.d_ff, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ShadowDropBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = ShadowDropAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = ShadowDropMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class ShadowDropConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.flip_rate = kwargs.get("flip_rate", 0.05)
        self.error_decay = kwargs.get("error_decay", 0.999)
        self.drop_step = kwargs.get("drop_step", 5000)


class ShadowDropTransformer(nn.Module):
    """Train with shadows (STE+Adam), then drop them mid-training.

    Phase 1 (steps 0 → drop_step): STE forward, Adam optimizes shadow weights
    Phase 2 (steps drop_step → end): DQT forward, discrete optimization only

    The transition seeds the DQT error buffer with the quantization residual,
    preserving as much shadow information as possible.
    """

    def __init__(self, cfg: ShadowDropConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = ShadowDropEmbedding(cfg.vocab_size, cfg.d_model, group_size=cfg.group_size,
                                          flip_rate=cfg.flip_rate, error_decay=cfg.error_decay)
        self.layers = nn.ModuleList([ShadowDropBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = ShadowDropLinear(cfg.d_model, cfg.vocab_size, group_size=cfg.group_size,
                                         flip_rate=cfg.flip_rate, error_decay=cfg.error_decay)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ShadowDropTransformer: {n_params/1e6:.1f}M params "
              f"(Phase 1: STE+Adam, drop shadows at step {cfg.drop_step})")

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict:
        B, S = input_ids.shape
        x = self.embed(input_ids)
        mask = self.causal_mask[:, :, :S, :S]

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    def maybe_drop_shadows(self, step: int):
        """Call each step. Transitions to DQT when step == drop_step."""
        if step != self.cfg.drop_step:
            return False

        print(f"\n  *** DROPPING SHADOW WEIGHTS at step {step} ***")
        for m in self.modules():
            if isinstance(m, (ShadowDropLinear, ShadowDropEmbedding)):
                m.transition_to_dqt()

        # Report memory savings
        n_ternary = sum(m.ternary.numel() for m in self.modules()
                        if hasattr(m, 'ternary') and m.ternary is not None)
        print(f"  Transitioned {n_ternary/1e6:.1f}M weights to DQT mode")
        print(f"  Shadow weights can now be freed by optimizer rebuild")
        return True

    def flip_step(self, scale_lr: float = 1e-3, temperature: float = 1.0) -> dict:
        """DQT updates — only active after shadow drop."""
        total = {"n_flips": 0, "n_sign": 0, "n_structural": 0}
        for m in self.modules():
            if isinstance(m, (ShadowDropLinear, ShadowDropEmbedding)):
                stats = m.dqt_step(temperature=temperature, scale_lr=scale_lr)
                total["n_flips"] += stats["n_flips"]
                total["n_sign"] += stats["n_sign"]
                total["n_structural"] += stats["n_structural"]
        return total

    def get_flip_params(self) -> list:
        """In phase 2, weight params should be excluded from optimizer."""
        if self.embed.phase == 2:
            return [m.weight for m in self.modules()
                    if isinstance(m, (ShadowDropLinear, ShadowDropEmbedding))]
        return []

    def ternary_stats(self) -> dict:
        ternary_tensors = [m.ternary.flatten().float()
                           for m in self.modules()
                           if hasattr(m, 'ternary') and m.ternary is not None]
        if not ternary_tensors:
            return {"neg_frac": 0.33, "zero_frac": 0.33, "pos_frac": 0.33}
        all_t = torch.cat(ternary_tensors)
        result = {
            "neg_frac": (all_t == -1).float().mean().item(),
            "zero_frac": (all_t == 0).float().mean().item(),
            "pos_frac": (all_t == 1).float().mean().item(),
        }
        error_tensors = [m.error.flatten().abs()
                         for m in self.modules()
                         if hasattr(m, 'error') and m.error is not None]
        if error_tensors:
            all_err = torch.cat(error_tensors)
            result["avg_error"] = all_err.mean().item()
            result["max_error"] = all_err.max().item()
        return result
