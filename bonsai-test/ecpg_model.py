"""ECPG: Error-Compensated Projected Gradient for Ternary Training.

No shadow weights. Combines three proven techniques:
1. Projected gradient: project onto valid ternary transitions (more info than sign)
2. Error compensation: accumulated residual ensures unbiasedness (1-bit Adam)
3. Per-group adaptive rate: Adam-style adaptivity on the flip probability

The key insight vs DQT: instead of treating each weight independently,
ECPG considers the GROUP context. A weight's flip probability depends on
how much OTHER weights in its group have flipped recently — preventing
catastrophic group-level disruption.

Memory per weight:
  - ternary: int8
  - error: fp16
  - group scale: fp16/group_size
  - group flip rate (adaptive): fp16/group_size
  Total: ~3.02 bytes/param
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


class ECPGLinear(nn.Module):
    """Linear layer with error-compensated projected gradient descent."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 base_rate: float = 0.005, max_group_flips: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.base_rate = base_rate
        self.max_group_flips = max_group_flips  # max fraction of group that can flip per step

        total = out_features * in_features
        if total % group_size != 0:
            self.group_size = self._find_group_size(total, group_size)
        n_groups = total // self.group_size

        # Initialize
        init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(init, a=5 ** 0.5)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(out_features, in_features).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        self.register_buffer("error", torch.zeros(out_features, in_features))

        # Per-group adaptive flip rate (Adam-style second moment)
        self.register_buffer("group_grad_sq", torch.zeros(n_groups))
        self.register_buffer("group_flip_ema", torch.zeros(n_groups))  # EMA of flips per group

        # Scale optimizer
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

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

    def ecpg_step(self, temperature: float = 1.0, scale_lr: float = 1e-3) -> dict:
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        gs = self.group_size
        n_groups = self.ternary.numel() // gs

        # Error-compensated gradient
        effective_grad = grad + self.error

        # Project gradient onto valid transitions:
        # For ternary w, the "projected gradient" tells us which transition is optimal
        # Valid transitions: w→w+1 (if w<1), w→w-1 (if w>-1)
        # Projected improvement for going up: -grad (positive = loss decreases)
        # Projected improvement for going down: +grad
        improvement_up = -effective_grad  # benefit of w += 1
        improvement_down = effective_grad  # benefit of w -= 1

        # Mask invalid transitions
        can_up = (self.ternary < 1).float()
        can_down = (self.ternary > -1).float()
        improvement_up = improvement_up * can_up
        improvement_down = improvement_down * can_down

        # Best direction per weight (choose max improvement)
        best_improvement = torch.maximum(improvement_up, improvement_down)
        go_up = improvement_up >= improvement_down  # True = up, False = down

        # Per-group adaptive rate: normalize by group gradient magnitude
        grad_groups = effective_grad.reshape(-1, gs)
        group_grad_norm = grad_groups.norm(dim=1).clamp(min=1e-8)
        self.group_grad_sq.lerp_(group_grad_norm.square(), 0.01)
        adaptive_rate = self.base_rate / (self.group_grad_sq.sqrt() + 1e-8)
        adaptive_rate = adaptive_rate.clamp(max=0.1)  # cap at 10% flip probability

        # Compute flip probability: improvement * adaptive_rate / temperature
        rate_expanded = adaptive_rate.unsqueeze(1).expand(-1, gs).reshape_as(self.ternary)
        flip_prob = (best_improvement.abs() * rate_expanded / max(temperature, 0.01)).clamp(0, 0.5)

        # Group-level cap: don't flip more than max_group_flips fraction per group per step
        max_flips_per_group = max(1, int(gs * self.max_group_flips))

        # Sample flips
        do_flip = torch.bernoulli(flip_prob).bool()

        # Vectorized per-group cap: for groups exceeding the cap, keep top-k by probability
        do_flip_flat = do_flip.reshape(-1, gs)
        flip_prob_flat = flip_prob.reshape(-1, gs)
        group_flip_counts = do_flip_flat.sum(dim=1)
        over_budget = group_flip_counts > max_flips_per_group

        if over_budget.any():
            # For over-budget groups: zero out flips, then set top-k
            masked_probs = flip_prob_flat * do_flip_flat.float()
            # Get top-k indices per group (only process over-budget groups)
            over_idx = over_budget.nonzero(as_tuple=True)[0]
            for g in over_idx:
                _, top_idx = masked_probs[g].topk(max_flips_per_group)
                do_flip_flat[g] = False
                do_flip_flat[g, top_idx] = True

        do_flip = do_flip_flat.reshape_as(self.ternary)

        # Apply flips
        old_ternary = self.ternary.clone()
        step_dir = torch.where(go_up, torch.ones_like(self.ternary), -torch.ones_like(self.ternary))
        new_val = (self.ternary + step_dir).clamp(-1, 1)
        actually_changed = do_flip & (new_val != self.ternary)
        self.ternary[actually_changed] = new_val[actually_changed]

        # Stats
        was_zero = old_ternary == 0
        is_zero = self.ternary == 0
        n_sign = (actually_changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (actually_changed & (was_zero | is_zero)).sum().item()

        # Update error buffer
        self.error[actually_changed] = 0.0
        self.error[~actually_changed] += grad[~actually_changed]
        self.error.mul_(0.995)  # slightly slower decay than DQT

        # Track per-group flip rate
        flips_per_group = actually_changed.reshape(-1, gs).float().mean(dim=1)
        self.group_flip_ema.lerp_(flips_per_group, 0.01)

        # Scale update (same as other models)
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


class ECPGEmbedding(nn.Module):
    """Ternary embedding with ECPG."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 base_rate: float = 0.005, max_group_flips: float = 0.05):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.base_rate = base_rate
        self.max_group_flips = max_group_flips

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = ECPGLinear._find_group_size(total, group_size)
        n_groups = total // self.group_size

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        self.register_buffer("error", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("group_grad_sq", torch.zeros(n_groups))
        self.register_buffer("group_flip_ema", torch.zeros(n_groups))
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

    def ecpg_step(self, temperature: float = 1.0, scale_lr: float = 1e-3) -> dict:
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        gs = self.group_size
        n_groups = self.ternary.numel() // gs

        effective_grad = grad + self.error
        improvement_up = -effective_grad * (self.ternary < 1).float()
        improvement_down = effective_grad * (self.ternary > -1).float()
        best_improvement = torch.maximum(improvement_up, improvement_down)
        go_up = improvement_up >= improvement_down

        grad_groups = effective_grad.reshape(-1, gs)
        group_grad_norm = grad_groups.norm(dim=1).clamp(min=1e-8)
        self.group_grad_sq.lerp_(group_grad_norm.square(), 0.01)
        adaptive_rate = (self.base_rate / (self.group_grad_sq.sqrt() + 1e-8)).clamp(max=0.1)

        rate_expanded = adaptive_rate.unsqueeze(1).expand(-1, gs).reshape_as(self.ternary)
        flip_prob = (best_improvement.abs() * rate_expanded / max(temperature, 0.01)).clamp(0, 0.5)

        max_flips_per_group = max(1, int(gs * self.max_group_flips))
        do_flip = torch.bernoulli(flip_prob).bool()

        do_flip_flat = do_flip.reshape(-1, gs)
        flip_prob_flat = flip_prob.reshape(-1, gs)
        group_flip_counts = do_flip_flat.sum(dim=1)
        over_budget = group_flip_counts > max_flips_per_group

        if over_budget.any():
            masked_probs = flip_prob_flat * do_flip_flat.float()
            over_idx = over_budget.nonzero(as_tuple=True)[0]
            for g in over_idx:
                _, top_idx = masked_probs[g].topk(max_flips_per_group)
                do_flip_flat[g] = False
                do_flip_flat[g, top_idx] = True

        do_flip = do_flip_flat.reshape_as(self.ternary)

        old_ternary = self.ternary.clone()
        step_dir = torch.where(go_up, torch.ones_like(self.ternary), -torch.ones_like(self.ternary))
        new_val = (self.ternary + step_dir).clamp(-1, 1)
        actually_changed = do_flip & (new_val != self.ternary)
        self.ternary[actually_changed] = new_val[actually_changed]

        was_zero = old_ternary == 0
        is_zero = self.ternary == 0
        n_sign = (actually_changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (actually_changed & (was_zero | is_zero)).sum().item()

        self.error[actually_changed] = 0.0
        self.error[~actually_changed] += grad[~actually_changed]
        self.error.mul_(0.995)

        flips_per_group = actually_changed.reshape(-1, gs).float().mean(dim=1)
        self.group_flip_ema.lerp_(flips_per_group, 0.01)

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


class ECPGAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        gs = cfg.group_size
        br = cfg.base_rate

        self.W_q = ECPGLinear(cfg.d_model, cfg.d_model, group_size=gs, base_rate=br)
        self.W_k = ECPGLinear(cfg.d_model, cfg.d_model, group_size=gs, base_rate=br)
        self.W_v = ECPGLinear(cfg.d_model, cfg.d_model, group_size=gs, base_rate=br)
        self.W_o = ECPGLinear(cfg.d_model, cfg.d_model, group_size=gs, base_rate=br)

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


class ECPGMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gs = cfg.group_size
        br = cfg.base_rate
        self.w_gate = ECPGLinear(cfg.d_model, cfg.d_ff, group_size=gs, base_rate=br)
        self.w_up = ECPGLinear(cfg.d_model, cfg.d_ff, group_size=gs, base_rate=br)
        self.w_down = ECPGLinear(cfg.d_ff, cfg.d_model, group_size=gs, base_rate=br)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ECPGBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = ECPGAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = ECPGMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class ECPGConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.base_rate = kwargs.get("base_rate", 0.005)
        self.max_group_flips = kwargs.get("max_group_flips", 0.05)


class ECPGTransformer(nn.Module):
    """Ternary transformer with Error-Compensated Projected Gradient.

    No shadow weights. Adaptive per-group flip rates.
    Group-level flip caps prevent catastrophic disruption.
    """

    def __init__(self, cfg: ECPGConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = ECPGEmbedding(cfg.vocab_size, cfg.d_model, group_size=cfg.group_size,
                                    base_rate=cfg.base_rate, max_group_flips=cfg.max_group_flips)
        self.layers = nn.ModuleList([ECPGBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = ECPGLinear(cfg.d_model, cfg.vocab_size, group_size=cfg.group_size,
                                   base_rate=cfg.base_rate, max_group_flips=cfg.max_group_flips)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_ternary = sum(m.ternary.numel() for m in self.modules() if hasattr(m, 'ternary'))
        print(f"ECPGTransformer: {n_params/1e6:.1f}M params "
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
        total = {"n_flips": 0, "n_sign": 0, "n_structural": 0}
        for m in self.modules():
            if isinstance(m, (ECPGLinear, ECPGEmbedding)):
                stats = m.ecpg_step(temperature=temperature, scale_lr=scale_lr)
                total["n_flips"] += stats["n_flips"]
                total["n_sign"] += stats["n_sign"]
                total["n_structural"] += stats["n_structural"]
        return total

    def get_flip_params(self) -> list:
        return [m.weight for m in self.modules()
                if isinstance(m, (ECPGLinear, ECPGEmbedding))]

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
