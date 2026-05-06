"""SPRT-Flip: ternary optimizer based on Sequential Probability Ratio Test.

Each weight is a state machine {-1, 0, +1} with transitions governed by
Wald's SPRT (1945) — the provably optimal sequential hypothesis test.

For each weight, we test:
  H0: weight is in the correct ternary state (gradient signs are noise)
  H1: weight should change (gradient signs are biased)

The log-likelihood ratio accumulates integer evidence. When it crosses
a threshold derived from the desired error rate, the weight flips.

No tunable hyperparameters beyond:
  - p_signal: gradient sign accuracy (default 0.6, estimable from data)
  - alpha: desired false-flip rate (default 0.01 = 99% confidence)
  - alpha_structural: error rate for 0↔±1 flips (default 0.001)

Everything else is derived from these via Wald's formulas.

Memory per weight: 2 × int16 (log_odds_up, log_odds_down) = 4 bytes
All operations: integer add, bit-shift, compare. No floating point.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


def compute_sprt_constants(p_signal: float = 0.6, alpha: float = 0.01,
                           alpha_structural: float = 0.001,
                           int_scale: int = 16):
    """Derive all SPRT constants from statistical parameters.

    Returns dict of integer constants for the kernel.
    """
    # Log-likelihood increments per observation
    c_pos = math.log(p_signal / 0.5)         # evidence FOR observed direction
    c_neg = math.log((1 - p_signal) / 0.5)   # evidence AGAINST

    # Thresholds from desired error rates (Wald's formula)
    tau_sign = math.log((1 - alpha) / alpha)
    tau_structural = math.log((1 - alpha_structural) / alpha_structural)

    # Steps needed for consistent signal to trigger flip
    steps_sign = tau_sign / c_pos
    steps_structural = tau_structural / c_pos

    # Scale to integer
    c_pos_int = max(1, round(c_pos * int_scale))
    c_neg_int = round(c_neg * int_scale)  # negative
    tau_sign_int = round(tau_sign * int_scale)
    tau_structural_int = round(tau_structural * int_scale)

    return {
        "c_pos": c_pos_int,
        "c_neg": c_neg_int,
        "tau_sign": tau_sign_int,
        "tau_structural": tau_structural_int,
        "int_scale": int_scale,
        # For logging
        "steps_to_sign_flip": steps_sign,
        "steps_to_structural_flip": steps_structural,
        "p_signal": p_signal,
        "alpha": alpha,
        "alpha_structural": alpha_structural,
    }


class SPRTFlipLinear(nn.Module):
    """Ternary linear with SPRT-based flip decisions.

    Each weight maintains two log-odds accumulators (int16):
      odds_up:   evidence that weight should increase
      odds_down: evidence that weight should decrease

    Transitions: -1 →(up)→ 0 →(up)→ +1
                 +1 →(down)→ 0 →(down)→ -1
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 sprt: dict | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        if sprt is None:
            sprt = compute_sprt_constants()
        self.c_pos = sprt["c_pos"]
        self.c_neg = sprt["c_neg"]
        self.tau_sign = sprt["tau_sign"]
        self.tau_structural = sprt["tau_structural"]

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

        # SPRT accumulators — int16, range ±32768
        self.register_buffer("odds_up", torch.zeros(out_features, in_features, dtype=torch.int16))
        self.register_buffer("odds_down", torch.zeros(out_features, in_features, dtype=torch.int16))

        # Scale optimizer (tiny Adam)
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        # Gradient capture (NOT in optimizer)
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

    def flip_step(self, scale_lr: float = 1e-3) -> dict:
        """SPRT update: accumulate evidence, flip on threshold crossing."""
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()

        # --- 1. Gradient sign → evidence update ---
        # Negative gradient = loss decreases if weight increases = evidence for UP
        grad_negative = grad < 0  # evidence for "should increase"
        grad_positive = grad > 0  # evidence for "should decrease"

        # Update odds_up: how much evidence that weight should be higher
        # grad negative → FOR up (add c_pos), grad positive → AGAINST up (add c_neg)
        up_delta = torch.where(grad_negative, self.c_pos,
                    torch.where(grad_positive, self.c_neg, 0)).to(torch.int16)
        self.odds_up.add_(up_delta)

        # Update odds_down: evidence that weight should be lower
        down_delta = torch.where(grad_positive, self.c_pos,
                      torch.where(grad_negative, self.c_neg, 0)).to(torch.int16)
        self.odds_down.add_(down_delta)

        # Floor at zero — negative log-odds means H0 is winning, reset
        self.odds_up.clamp_(min=0)
        self.odds_down.clamp_(min=0)

        # --- 2. Non-stationarity decay (bit-shift) ---
        # odds -= odds >> 10  ≈ multiply by (1 - 1/1024) ≈ 0.999 per step
        self.odds_up.sub_(self.odds_up >> 10)
        self.odds_down.sub_(self.odds_down >> 10)

        # --- 3. State-dependent threshold ---
        is_at_zero = self.ternary == 0
        # Flipping FROM zero = structural (add connection) → higher threshold
        # Flipping TO a different sign = sign flip → lower threshold
        tau_up = torch.where(is_at_zero, self.tau_structural, self.tau_sign)
        tau_down = torch.where(is_at_zero, self.tau_structural, self.tau_sign)

        # --- 4. Threshold crossing → flip ---
        do_up = (self.odds_up > tau_up) & (self.ternary < 1)
        do_down = (self.odds_down > tau_down) & (self.ternary > -1)

        # If both want to flip (shouldn't happen often), pick stronger evidence
        both = do_up & do_down
        if both.any():
            prefer_up = self.odds_up >= self.odds_down
            do_up = do_up & (~both | prefer_up)
            do_down = do_down & (~both | ~prefer_up)

        is_structural = (self.ternary == 0) & (do_up | do_down)
        n_structural = is_structural.sum().item()
        n_sign = (do_up | do_down).sum().item() - n_structural

        # --- 5. Apply flips + reset accumulators ---
        if n_sign + n_structural > 0:
            self.ternary[do_up] += 1
            self.ternary[do_down] -= 1

        # Reset BOTH accumulators for flipped weights (fresh evidence in new state)
        flipped = do_up | do_down
        self.odds_up[flipped] = 0
        self.odds_down[flipped] = 0

        # --- 6. Update group scales ---
        self.scale_step += 1
        gs = self.group_size
        grad_scale = (grad * self.ternary.float()).reshape(-1, gs).sum(dim=1)
        b1, b2 = 0.9, 0.999
        self.scale_m1.lerp_(grad_scale, 1 - b1)
        self.scale_m2.lerp_(grad_scale.square(), 1 - b2)
        m1_hat = self.scale_m1 / (1 - b1 ** self.scale_step)
        m2_hat = self.scale_m2 / (1 - b2 ** self.scale_step)
        self.scales.addcdiv_(m1_hat, m2_hat.sqrt().add_(1e-8), value=-scale_lr)
        self.scales.clamp_(min=1e-6)

        # --- 7. Clear gradient ---
        self.weight.grad = None

        return {"n_flips": n_sign + n_structural, "n_sign": n_sign, "n_structural": n_structural}


class SPRTFlipEmbedding(nn.Module):
    """Ternary embedding with SPRT-based flip decisions."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 sprt: dict | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        if sprt is None:
            sprt = compute_sprt_constants()
        self.c_pos = sprt["c_pos"]
        self.c_neg = sprt["c_neg"]
        self.tau_sign = sprt["tau_sign"]
        self.tau_structural = sprt["tau_structural"]

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = SPRTFlipLinear._find_group_size(total, group_size)
        n_groups = total // self.group_size

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        self.register_buffer("odds_up", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int16))
        self.register_buffer("odds_down", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int16))
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

    def flip_step(self, scale_lr: float = 1e-3) -> dict:
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        grad_negative = grad < 0
        grad_positive = grad > 0

        up_delta = torch.where(grad_negative, self.c_pos,
                    torch.where(grad_positive, self.c_neg, 0)).to(torch.int16)
        down_delta = torch.where(grad_positive, self.c_pos,
                      torch.where(grad_negative, self.c_neg, 0)).to(torch.int16)
        self.odds_up.add_(up_delta).clamp_(min=0)
        self.odds_down.add_(down_delta).clamp_(min=0)

        self.odds_up.sub_(self.odds_up >> 10)
        self.odds_down.sub_(self.odds_down >> 10)

        is_at_zero = self.ternary == 0
        tau_up = torch.where(is_at_zero, self.tau_structural, self.tau_sign)
        tau_down = torch.where(is_at_zero, self.tau_structural, self.tau_sign)

        do_up = (self.odds_up > tau_up) & (self.ternary < 1)
        do_down = (self.odds_down > tau_down) & (self.ternary > -1)

        both = do_up & do_down
        if both.any():
            prefer_up = self.odds_up >= self.odds_down
            do_up = do_up & (~both | prefer_up)
            do_down = do_down & (~both | ~prefer_up)

        is_structural = (self.ternary == 0) & (do_up | do_down)
        n_structural = is_structural.sum().item()
        n_sign = (do_up | do_down).sum().item() - n_structural

        if n_sign + n_structural > 0:
            self.ternary[do_up] += 1
            self.ternary[do_down] -= 1

        flipped = do_up | do_down
        self.odds_up[flipped] = 0
        self.odds_down[flipped] = 0

        self.scale_step += 1
        gs = self.group_size
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


class SPRTFlipAttention(nn.Module):
    def __init__(self, cfg, sprt):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = SPRTFlipLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)
        self.W_k = SPRTFlipLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)
        self.W_v = SPRTFlipLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)
        self.W_o = SPRTFlipLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)

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

        return self.W_o(out.transpose(1, 2).reshape(B, S, D))


class SPRTFlipMLP(nn.Module):
    def __init__(self, cfg, sprt):
        super().__init__()
        self.w_gate = SPRTFlipLinear(cfg.d_model, cfg.d_ff, cfg.group_size, sprt)
        self.w_up = SPRTFlipLinear(cfg.d_model, cfg.d_ff, cfg.group_size, sprt)
        self.w_down = SPRTFlipLinear(cfg.d_ff, cfg.d_model, cfg.group_size, sprt)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SPRTFlipBlock(nn.Module):
    def __init__(self, cfg, sprt):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = SPRTFlipAttention(cfg, sprt)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = SPRTFlipMLP(cfg, sprt)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class SPRTFlipConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.p_signal = kwargs.get("p_signal", 0.6)
        self.alpha = kwargs.get("alpha", 0.01)
        self.alpha_structural = kwargs.get("alpha_structural", 0.001)
        self.attn_residual = kwargs.get("attn_residual", True)
        self.attn_residual_init = kwargs.get("attn_residual_init", 0.1)


class SPRTFlipTransformer(nn.Module):
    """Ternary transformer with SPRT-based statistically principled flip decisions."""

    def __init__(self, cfg: SPRTFlipConfig):
        super().__init__()
        self.cfg = cfg
        sprt = compute_sprt_constants(cfg.p_signal, cfg.alpha, cfg.alpha_structural)

        print(f"SPRT constants: c+={sprt['c_pos']}, c-={sprt['c_neg']}, "
              f"τ_sign={sprt['tau_sign']}, τ_struct={sprt['tau_structural']}")
        print(f"  Steps to sign flip: {sprt['steps_to_sign_flip']:.0f} consistent signals")
        print(f"  Steps to structural flip: {sprt['steps_to_structural_flip']:.0f} consistent signals")

        self.embed = SPRTFlipEmbedding(cfg.vocab_size, cfg.d_model, cfg.group_size, sprt)
        self.layers = nn.ModuleList([SPRTFlipBlock(cfg, sprt) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = SPRTFlipLinear(cfg.d_model, cfg.vocab_size, cfg.group_size, sprt)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_ternary = sum(m.ternary.numel() for m in self.modules() if hasattr(m, 'ternary'))
        optimizer_bytes = sum(m.odds_up.numel() * 2 + m.odds_down.numel() * 2
                              for m in self.modules() if hasattr(m, 'odds_up'))
        print(f"SPRTFlipTransformer: {n_params/1e6:.1f}M params "
              f"({n_ternary/1e6:.1f}M ternary)")
        print(f"  Optimizer state: {optimizer_bytes/1e6:.1f}MB "
              f"({optimizer_bytes/n_ternary:.1f} bytes/weight)")

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
            if isinstance(m, (SPRTFlipLinear, SPRTFlipEmbedding)):
                stats = m.flip_step(scale_lr=scale_lr)
                for k in totals:
                    totals[k] += stats[k]
        return totals

    def get_flip_params(self) -> list:
        return [m.weight for m in self.modules()
                if isinstance(m, (SPRTFlipLinear, SPRTFlipEmbedding))]

    def ternary_stats(self) -> dict:
        all_t = torch.cat([m.ternary.flatten().float()
                           for m in self.modules() if hasattr(m, 'ternary')])
        return {
            "neg_frac": (all_t == -1).float().mean().item(),
            "zero_frac": (all_t == 0).float().mean().item(),
            "pos_frac": (all_t == 1).float().mean().item(),
        }
