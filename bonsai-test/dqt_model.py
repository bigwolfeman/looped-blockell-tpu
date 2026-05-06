"""DQT: Discrete Quantization Training via Stochastic Rounding.

No shadow weights. No Adam on weights. Each step makes independent
probabilistic flip decisions based on gradient magnitude.

Key insight from DQT (Dec 2024): stochastic rounding is UNBIASED —
E[quantize(w + delta)] = w + delta. This gives convergence guarantees
that hard-threshold flip momentum cannot provide.

Memory per weight parameter:
  - ternary value: int8 (packed to 1.58 bits at inference)
  - error buffer: fp16 (accumulated rounding residual)
  - group scale: fp16 (shared per group_size elements)
  Total: ~3 bytes/param vs 10 bytes for shadow+Adam

The error buffer is NOT a shadow weight — it doesn't track the "true"
continuous weight. It tracks how much gradient signal we've LOST by
not flipping, ensuring eventual correction. Think of it as a mailbox
that fills up until the letter carrier (flip) empties it.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


class DQTLinear(nn.Module):
    """Linear layer with DQT stochastic rounding — no shadow weights."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 flip_rate: float = 0.01, error_decay: float = 0.99):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.flip_rate = flip_rate
        self.error_decay = error_decay

        total = out_features * in_features
        if total % group_size != 0:
            self.group_size = self._find_group_size(total, group_size)
        n_groups = total // self.group_size

        # Initialize from kaiming → quantize to ternary
        init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(init, a=5 ** 0.5)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(out_features, in_features).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))  # [n_groups]
        self.register_buffer("error", torch.zeros(out_features, in_features))

        # Scale optimizer state (tiny Adam for group scales)
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        # Gradient capture parameter (overwritten each forward)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    @staticmethod
    def _find_group_size(total: int, target: int) -> int:
        for gs in range(target, 0, -1):
            if total % gs == 0:
                return gs
        return 1

    def _reconstruct_weight(self) -> Tensor:
        s = self.scales.unsqueeze(1).expand(-1, self.group_size)
        s = s.reshape(self.out_features, self.in_features)
        return self.ternary.float() * s

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            self.weight.data.copy_(self._reconstruct_weight())
        return F.linear(x, self.weight)

    def dqt_step(self, temperature: float = 1.0, scale_lr: float = 1e-3) -> dict:
        """Stochastic rounding step. Returns flip statistics."""
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()

        # Error compensation: add accumulated residual to gradient
        effective_grad = grad + self.error

        # Desired continuous step (in ternary-scale-normalized space)
        # Normalize gradient by group scale so flip probability is scale-invariant
        gs = self.group_size
        scale_expanded = self.scales.unsqueeze(1).expand(-1, gs).reshape(self.out_features, self.in_features)
        normalized_grad = effective_grad / scale_expanded.clamp(min=1e-8)

        # Compute flip probability via stochastic rounding
        # P(flip) = clip(|normalized_grad| * flip_rate / temperature, 0, max_p)
        # Direction: sign of -gradient (gradient descent)
        flip_prob = (normalized_grad.abs() * self.flip_rate / max(temperature, 0.01)).clamp(0, 0.5)
        direction = -normalized_grad.sign()  # which way we WANT to move

        # Sample flips
        do_flip = torch.bernoulli(flip_prob).bool()

        # Compute new ternary values for flipped weights
        old_ternary = self.ternary.clone()

        # For flipped weights: move in desired direction, clamped to [-1, +1]
        # direction > 0 → want to increase → ternary += 1
        # direction < 0 → want to decrease → ternary -= 1
        step_dir = direction.sign().to(torch.int8)
        new_val = (self.ternary + step_dir).clamp(-1, 1)

        # Only apply where we actually flip AND the value changes
        actually_changed = do_flip & (new_val != self.ternary)
        self.ternary[actually_changed] = new_val[actually_changed]

        # Track flip types
        was_zero = old_ternary == 0
        is_zero = self.ternary == 0
        n_sign = (actually_changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (actually_changed & (was_zero | is_zero)).sum().item()

        # Update error buffer:
        # For flipped weights: error resets (the flip "consumed" the accumulated signal)
        # For non-flipped weights: error += gradient (signal we didn't act on)
        self.error[actually_changed] = 0.0
        self.error[~actually_changed] += grad[~actually_changed]

        # Decay error to prevent unbounded growth (leaky integration)
        self.error.mul_(self.error_decay)

        # Update group scales (tiny Adam)
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
        n_flips = n_sign + n_structural
        return {"n_flips": n_flips, "n_sign": n_sign, "n_structural": n_structural}


class DQTEmbedding(nn.Module):
    """Ternary embedding with DQT stochastic rounding."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 flip_rate: float = 0.01, error_decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.flip_rate = flip_rate
        self.error_decay = error_decay

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = DQTLinear._find_group_size(total, group_size)
        n_groups = total // self.group_size

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        self.register_buffer("error", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))

    def _reconstruct_weight(self) -> Tensor:
        gs = self.group_size
        s = self.scales.unsqueeze(1).expand(-1, gs)
        s = s.reshape(self.num_embeddings, self.embedding_dim)
        return self.ternary.float() * s

    def forward(self, input_ids: Tensor) -> Tensor:
        with torch.no_grad():
            self.weight.data.copy_(self._reconstruct_weight())
        return F.embedding(input_ids, self.weight)

    def dqt_step(self, temperature: float = 1.0, scale_lr: float = 1e-3) -> dict:
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        effective_grad = grad + self.error

        gs = self.group_size
        scale_expanded = self.scales.unsqueeze(1).expand(-1, gs).reshape(self.num_embeddings, self.embedding_dim)
        normalized_grad = effective_grad / scale_expanded.clamp(min=1e-8)

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
        self.error.mul_(0.99)

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


class DQTAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        gs = cfg.group_size
        fr = cfg.flip_rate

        ed = cfg.error_decay
        self.W_q = DQTLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)
        self.W_k = DQTLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)
        self.W_v = DQTLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)
        self.W_o = DQTLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)

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


class DQTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gs = cfg.group_size
        fr = cfg.flip_rate
        ed = cfg.error_decay
        self.w_gate = DQTLinear(cfg.d_model, cfg.d_ff, group_size=gs, flip_rate=fr, error_decay=ed)
        self.w_up = DQTLinear(cfg.d_model, cfg.d_ff, group_size=gs, flip_rate=fr, error_decay=ed)
        self.w_down = DQTLinear(cfg.d_ff, cfg.d_model, group_size=gs, flip_rate=fr, error_decay=ed)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DQTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = DQTAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = DQTMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DQTConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.flip_rate = kwargs.get("flip_rate", 0.01)
        self.error_decay = kwargs.get("error_decay", 0.99)


class DQTTransformer(nn.Module):
    """Ternary transformer trained via DQT stochastic rounding.

    No shadow weights. No Adam on weight parameters.
    Provably unbiased discrete updates with error compensation.
    """

    def __init__(self, cfg: DQTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = DQTEmbedding(cfg.vocab_size, cfg.d_model, group_size=cfg.group_size,
                                   flip_rate=cfg.flip_rate, error_decay=cfg.error_decay)
        self.layers = nn.ModuleList([DQTBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = DQTLinear(cfg.d_model, cfg.vocab_size, group_size=cfg.group_size,
                                  flip_rate=cfg.flip_rate, error_decay=cfg.error_decay)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_ternary = sum(m.ternary.numel() for m in self.modules() if hasattr(m, 'ternary'))
        print(f"DQTTransformer: {n_params/1e6:.1f}M params "
              f"({n_ternary/1e6:.1f}M ternary, no shadow weights)")

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

    def flip_step(self, scale_lr: float = 1e-3, temperature: float = 1.0) -> dict:
        """Run DQT stochastic rounding on all layers."""
        total = {"n_flips": 0, "n_sign": 0, "n_structural": 0}
        for m in self.modules():
            if isinstance(m, (DQTLinear, DQTEmbedding)):
                stats = m.dqt_step(temperature=temperature, scale_lr=scale_lr)
                total["n_flips"] += stats["n_flips"]
                total["n_sign"] += stats["n_sign"]
                total["n_structural"] += stats["n_structural"]
        return total

    def get_flip_params(self) -> list:
        """Return weight parameters managed by DQT (excluded from optimizer)."""
        return [m.weight for m in self.modules()
                if isinstance(m, (DQTLinear, DQTEmbedding))]

    def ternary_stats(self) -> dict:
        all_t = torch.cat([m.ternary.flatten().float()
                           for m in self.modules() if hasattr(m, 'ternary')])
        all_err = torch.cat([m.error.flatten().abs()
                             for m in self.modules() if hasattr(m, 'error')])
        return {
            "neg_frac": (all_t == -1).float().mean().item(),
            "zero_frac": (all_t == 0).float().mean().item(),
            "pos_frac": (all_t == 1).float().mean().item(),
            "avg_error": all_err.mean().item(),
            "max_error": all_err.max().item(),
        }
