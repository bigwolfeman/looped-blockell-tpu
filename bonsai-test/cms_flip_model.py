"""CMS-Flip: ternary optimizer with CMS-style multi-timescale scoring.

Combines our proven CMS gradient EMA infrastructure with Bop-style
discrete flip decisions. Per-weight, multi-timescale evidence accumulation
with adaptive state-dependent thresholds.

Key improvements over naive flip_model.py:
  1. Multi-timescale EMA (fast γ=0.9, slow γ=0.999) — flip only when BOTH agree
  2. Per-weight adaptive threshold from flip history (frequent flippers need more evidence)
  3. State-dependent thresholds (0↔±1 structural > +1↔-1 sign flips)
  4. Gradient magnitude weighting (large gradients = stronger evidence)
  5. Stochastic threshold near boundary for unbiasedness

Memory per weight:
  - ternary: int8
  - ema_fast: fp16
  - ema_slow: fp16
  - flip_count: int16
  - group scale: fp16 / 128
  Total: ~5.2 bytes/param (vs 10 for shadow+Adam, vs 3.2 for naive flip)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


class CMSFlipLinear(nn.Module):
    """Ternary linear with CMS-style multi-timescale flip scoring."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 tau_base: float = 0.5, tau_structural: float = 1.5,
                 gamma_fast: float = 0.9, gamma_slow: float = 0.999):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.tau_base = tau_base
        self.tau_structural = tau_structural
        self.gamma_fast = gamma_fast
        self.gamma_slow = gamma_slow

        total = out_features * in_features
        if total % group_size != 0:
            self.group_size = self._find_group_size(total, group_size)
        n_groups = total // self.group_size

        # Initialize from kaiming → ternary
        init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(init, a=5 ** 0.5)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        # Core state
        self.register_buffer("ternary", ternary.reshape(out_features, in_features).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))

        # Multi-timescale EMA (CMS-style per-weight scoring)
        # Fast: reacts to recent gradients (~10 step window)
        # Slow: long-term consensus (~1000 step window)
        self.register_buffer("ema_fast", torch.zeros(out_features, in_features))
        self.register_buffer("ema_slow", torch.zeros(out_features, in_features))

        # Per-weight flip history for adaptive thresholds
        self.register_buffer("flip_count", torch.zeros(out_features, in_features, dtype=torch.int16))

        # Scale optimizer (tiny Adam)
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0
        self.global_step = 0

        # Gradient capture param (NOT in optimizer)
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

    def _compute_thresholds(self) -> Tensor:
        """Per-weight adaptive threshold.

        Three factors:
          1. Base threshold (tau_base for sign flips, tau_structural for 0↔±1)
          2. Flip-history penalty: frequent flippers need more evidence
          3. Global annealing: thresholds increase over training (crystallization)
        """
        # State-dependent: structural flips (involving zero) need higher threshold
        is_structural = (self.ternary == 0)  # flipping FROM zero = adding a connection
        tau = torch.where(is_structural,
                          torch.tensor(self.tau_structural, device=self.ternary.device),
                          torch.tensor(self.tau_base, device=self.ternary.device))

        # Flip-history penalty: sqrt(flip_count) scales threshold up
        tau = tau * (1.0 + 0.1 * self.flip_count.float().sqrt())

        # Global annealing: threshold grows with log(step) — explore early, crystallize late
        if self.global_step > 0:
            tau = tau * (1.0 + 0.1 * math.log1p(self.global_step / 1000))

        return tau

    def flip_step(self, scale_lr: float = 1e-3) -> dict:
        """CMS-style flip update. Returns stats dict."""
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        self.global_step += 1

        # --- 1. Update multi-timescale EMAs ---
        # Gradient signal: -sign(grad) * clipped_magnitude
        # Magnitude weighting: large gradients contribute more evidence
        grad_mag = grad.abs()
        grad_scale = grad_mag.mean().clamp(min=1e-8)
        signal = -grad.sign() * (grad_mag / grad_scale).clamp(max=1.0)

        self.ema_fast.lerp_(signal, 1 - self.gamma_fast)
        self.ema_slow.lerp_(signal, 1 - self.gamma_slow)

        # --- 2. Consensus check: both timescales must agree ---
        agree_up = (self.ema_fast > 0) & (self.ema_slow > 0)
        agree_down = (self.ema_fast < 0) & (self.ema_slow < 0)

        # Use the SLOWER EMA's magnitude as the evidence strength
        evidence = self.ema_slow.abs()

        # --- 3. Adaptive per-weight thresholds ---
        tau = self._compute_thresholds()

        # --- 4. Stochastic threshold for unbiasedness ---
        # Near the threshold: P(flip) = sigmoid((evidence - tau) * sharpness)
        # Far above: flip deterministically. Far below: don't flip.
        sharpness = 10.0
        p_flip = torch.sigmoid((evidence - tau) * sharpness)
        should_flip = torch.bernoulli(p_flip).bool()

        # --- 5. Determine flip direction and validity ---
        want_up = agree_up & should_flip
        want_down = agree_down & should_flip
        can_up = self.ternary < 1
        can_down = self.ternary > -1

        do_up = want_up & can_up
        do_down = want_down & can_down

        # Track structural vs sign flips
        is_structural_flip = (self.ternary == 0) & (do_up | do_down)
        n_structural = is_structural_flip.sum().item()
        n_sign = (do_up | do_down).sum().item() - n_structural

        # --- 6. Apply flips ---
        if n_sign + n_structural > 0:
            self.ternary[do_up] += 1
            self.ternary[do_down] -= 1
            self.flip_count[do_up | do_down] += 1

        # Reset EMAs for flipped weights (fresh start in new state)
        flipped = do_up | do_down
        self.ema_fast[flipped] = 0.0
        self.ema_slow[flipped] = 0.0

        # Also reset EMAs for boundary-saturated weights
        saturated = (want_up & ~can_up) | (want_down & ~can_down)
        self.ema_fast[saturated] = 0.0
        self.ema_slow[saturated] = 0.0

        # --- 7. Update group scales ---
        self.scale_step += 1
        gs = self.group_size
        grad_scale_vec = (grad * self.ternary.float()).reshape(-1, gs).sum(dim=1)
        b1, b2 = 0.9, 0.999
        self.scale_m1.lerp_(grad_scale_vec, 1 - b1)
        self.scale_m2.lerp_(grad_scale_vec.square(), 1 - b2)
        m1_hat = self.scale_m1 / (1 - b1 ** self.scale_step)
        m2_hat = self.scale_m2 / (1 - b2 ** self.scale_step)
        self.scales.addcdiv_(m1_hat, m2_hat.sqrt().add_(1e-8), value=-scale_lr)
        self.scales.clamp_(min=1e-6)

        # --- 8. Clear gradient ---
        self.weight.grad = None

        return {
            "n_flips": n_sign + n_structural,
            "n_sign": n_sign,
            "n_structural": n_structural,
        }


class CMSFlipEmbedding(nn.Module):
    """Ternary embedding with CMS-style flip scoring."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 tau_base: float = 0.5, tau_structural: float = 1.5,
                 gamma_fast: float = 0.9, gamma_slow: float = 0.999):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.tau_base = tau_base
        self.tau_structural = tau_structural
        self.gamma_fast = gamma_fast
        self.gamma_slow = gamma_slow

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = CMSFlipLinear._find_group_size(total, group_size)
        n_groups = total // self.group_size

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        self.register_buffer("ema_fast", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("ema_slow", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("flip_count", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int16))
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0
        self.global_step = 0

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

    def flip_step(self, scale_lr: float = 1e-3) -> dict:
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        self.global_step += 1

        grad_mag = grad.abs()
        grad_scale = grad_mag.mean().clamp(min=1e-8)
        signal = -grad.sign() * (grad_mag / grad_scale).clamp(max=1.0)

        self.ema_fast.lerp_(signal, 1 - self.gamma_fast)
        self.ema_slow.lerp_(signal, 1 - self.gamma_slow)

        agree_up = (self.ema_fast > 0) & (self.ema_slow > 0)
        agree_down = (self.ema_fast < 0) & (self.ema_slow < 0)
        evidence = self.ema_slow.abs()

        # Adaptive threshold
        is_structural = (self.ternary == 0)
        tau = torch.where(is_structural,
                          torch.tensor(self.tau_structural, device=self.ternary.device),
                          torch.tensor(self.tau_base, device=self.ternary.device))
        tau = tau * (1.0 + 0.1 * self.flip_count.float().sqrt())
        if self.global_step > 0:
            tau = tau * (1.0 + 0.1 * math.log1p(self.global_step / 1000))

        p_flip = torch.sigmoid((evidence - tau) * 10.0)
        should_flip = torch.bernoulli(p_flip).bool()

        want_up = agree_up & should_flip
        want_down = agree_down & should_flip
        do_up = want_up & (self.ternary < 1)
        do_down = want_down & (self.ternary > -1)

        is_structural_flip = (self.ternary == 0) & (do_up | do_down)
        n_structural = is_structural_flip.sum().item()
        n_sign = (do_up | do_down).sum().item() - n_structural

        if n_sign + n_structural > 0:
            self.ternary[do_up] += 1
            self.ternary[do_down] -= 1
            self.flip_count[do_up | do_down] += 1

        flipped = do_up | do_down
        saturated = (want_up & (self.ternary >= 1)) | (want_down & (self.ternary <= -1))
        self.ema_fast[flipped | saturated] = 0.0
        self.ema_slow[flipped | saturated] = 0.0

        # Scale update
        self.scale_step += 1
        gs = self.group_size
        grad_scale_vec = (grad * self.ternary.float()).reshape(-1, gs).sum(dim=1)
        b1, b2 = 0.9, 0.999
        self.scale_m1.lerp_(grad_scale_vec, 1 - b1)
        self.scale_m2.lerp_(grad_scale_vec.square(), 1 - b2)
        m1_hat = self.scale_m1 / (1 - b1 ** self.scale_step)
        m2_hat = self.scale_m2 / (1 - b2 ** self.scale_step)
        self.scales.addcdiv_(m1_hat, m2_hat.sqrt().add_(1e-8), value=-scale_lr)
        self.scales.clamp_(min=1e-6)

        self.weight.grad = None
        return {"n_flips": n_sign + n_structural, "n_sign": n_sign, "n_structural": n_structural}


class CMSFlipAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        kw = dict(group_size=cfg.group_size, tau_base=cfg.tau_base,
                  tau_structural=cfg.tau_structural,
                  gamma_fast=cfg.gamma_fast, gamma_slow=cfg.gamma_slow)
        self.W_q = CMSFlipLinear(cfg.d_model, cfg.d_model, **kw)
        self.W_k = CMSFlipLinear(cfg.d_model, cfg.d_model, **kw)
        self.W_v = CMSFlipLinear(cfg.d_model, cfg.d_model, **kw)
        self.W_o = CMSFlipLinear(cfg.d_model, cfg.d_model, **kw)

        if cfg.attn_residual:
            self.alpha = nn.Parameter(
                torch.full((cfg.n_heads, 1, 1), cfg.attn_residual_init))
        else:
            self.alpha = None

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

        if self.alpha is not None:
            x_heads = x.reshape(B, S, H, Dh).transpose(1, 2)
            out = out + self.alpha * x_heads

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_o(out)


class CMSFlipMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        kw = dict(group_size=cfg.group_size, tau_base=cfg.tau_base,
                  tau_structural=cfg.tau_structural,
                  gamma_fast=cfg.gamma_fast, gamma_slow=cfg.gamma_slow)
        self.w_gate = CMSFlipLinear(cfg.d_model, cfg.d_ff, **kw)
        self.w_up = CMSFlipLinear(cfg.d_model, cfg.d_ff, **kw)
        self.w_down = CMSFlipLinear(cfg.d_ff, cfg.d_model, **kw)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CMSFlipBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = CMSFlipAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = CMSFlipMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class CMSFlipConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.tau_base = kwargs.get("tau_base", 0.5)
        self.tau_structural = kwargs.get("tau_structural", 1.5)
        self.gamma_fast = kwargs.get("gamma_fast", 0.9)
        self.gamma_slow = kwargs.get("gamma_slow", 0.999)
        self.attn_residual = kwargs.get("attn_residual", True)
        self.attn_residual_init = kwargs.get("attn_residual_init", 0.1)


class CMSFlipTransformer(nn.Module):
    """Ternary transformer with CMS-style multi-timescale flip scoring."""

    def __init__(self, cfg: CMSFlipConfig):
        super().__init__()
        self.cfg = cfg

        kw = dict(group_size=cfg.group_size, tau_base=cfg.tau_base,
                  tau_structural=cfg.tau_structural,
                  gamma_fast=cfg.gamma_fast, gamma_slow=cfg.gamma_slow)
        self.embed = CMSFlipEmbedding(cfg.vocab_size, cfg.d_model, **kw)
        self.layers = nn.ModuleList([CMSFlipBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = CMSFlipLinear(cfg.d_model, cfg.vocab_size, **kw)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_ternary = sum(m.ternary.numel() for m in self.modules() if hasattr(m, 'ternary'))
        print(f"CMSFlipTransformer: {n_params/1e6:.1f}M params "
              f"({n_ternary/1e6:.1f}M ternary, CMS-style scoring)")

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
            out["loss"] = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), labels.view(-1), ignore_index=-100)
        return out

    def flip_step(self, scale_lr: float = 1e-3) -> dict:
        totals = {"n_flips": 0, "n_sign": 0, "n_structural": 0}
        for m in self.modules():
            if isinstance(m, (CMSFlipLinear, CMSFlipEmbedding)):
                stats = m.flip_step(scale_lr=scale_lr)
                for k in totals:
                    totals[k] += stats[k]
        return totals

    def get_flip_params(self) -> list:
        return [m.weight for m in self.modules()
                if isinstance(m, (CMSFlipLinear, CMSFlipEmbedding))]

    def ternary_stats(self) -> dict:
        all_t = torch.cat([m.ternary.flatten().float()
                           for m in self.modules() if hasattr(m, 'ternary')])
        all_fc = torch.cat([m.flip_count.flatten().float()
                            for m in self.modules() if hasattr(m, 'flip_count')])
        return {
            "neg_frac": (all_t == -1).float().mean().item(),
            "zero_frac": (all_t == 0).float().mean().item(),
            "pos_frac": (all_t == 1).float().mean().item(),
            "avg_flip_count": all_fc.mean().item(),
            "max_flip_count": all_fc.max().item(),
        }
