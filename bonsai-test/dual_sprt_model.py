"""Dual-SPRT: two independent state machines per weight.

Machine 1 — PARTICIPATION (magnitude-driven):
  Tests whether this weight should be active or zero.
  Evidence: gradient magnitude relative to layer mean.
  Large gradient = important = stay active / become active.
  Small gradient = irrelevant = stay zero / become zero.

Machine 2 — SIGN (sign-driven, active weights only):
  Tests whether active weight has correct sign.
  Evidence: gradient sign (standard SPRT).

Both machines use Wald's SPRT with integer arithmetic.

Memory per weight: 3 × int16 = 6 bytes
  - participation_evidence: int16
  - sign_evidence_up: int16
  - sign_evidence_down: int16
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


def compute_dual_sprt_constants(
    p_sign: float = 0.6,
    p_mag: float = 0.6,
    alpha_sign: float = 0.01,
    alpha_participation: float = 0.005,
    int_scale: int = 16,
):
    c_sign_pos = max(1, round(math.log(p_sign / 0.5) * int_scale))
    c_sign_neg = round(math.log((1 - p_sign) / 0.5) * int_scale)
    tau_sign = round(math.log((1 - alpha_sign) / alpha_sign) * int_scale)

    c_mag_pos = max(1, round(math.log(p_mag / 0.5) * int_scale))
    c_mag_neg = round(math.log((1 - p_mag) / 0.5) * int_scale)
    tau_participation = round(math.log((1 - alpha_participation) / alpha_participation) * int_scale)

    print(f"Dual-SPRT constants:")
    print(f"  Sign:  c+={c_sign_pos}, c-={c_sign_neg}, τ={tau_sign} "
          f"({tau_sign/c_sign_pos:.0f} consistent steps)")
    print(f"  Participation: c+={c_mag_pos}, c-={c_mag_neg}, τ={tau_participation} "
          f"({tau_participation/c_mag_pos:.0f} consistent steps)")

    return {
        "c_sign_pos": c_sign_pos, "c_sign_neg": c_sign_neg, "tau_sign": tau_sign,
        "c_mag_pos": c_mag_pos, "c_mag_neg": c_mag_neg, "tau_participation": tau_participation,
    }


class DualSPRTLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 sprt: dict | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        if sprt is None:
            sprt = compute_dual_sprt_constants()
        self.c_sign_pos = sprt["c_sign_pos"]
        self.c_sign_neg = sprt["c_sign_neg"]
        self.tau_sign = sprt["tau_sign"]
        self.c_mag_pos = sprt["c_mag_pos"]
        self.c_mag_neg = sprt["c_mag_neg"]
        self.tau_participation = sprt["tau_participation"]

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

        # Machine 1: participation evidence (magnitude-based)
        self.register_buffer("part_evidence", torch.zeros(out_features, in_features, dtype=torch.int16))

        # Machine 2: sign evidence (sign-based)
        self.register_buffer("sign_up", torch.zeros(out_features, in_features, dtype=torch.int16))
        self.register_buffer("sign_down", torch.zeros(out_features, in_features, dtype=torch.int16))

        # Scale optimizer
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        # Gradient capture
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
        if self.weight.grad is None:
            return {"n_activate": 0, "n_deactivate": 0, "n_sign": 0}

        grad = self.weight.grad.detach()
        grad_abs = grad.abs()

        # Layer-relative magnitude: is this gradient large or small?
        layer_mean = grad_abs.mean().clamp(min=1e-8)
        is_large = grad_abs > layer_mean  # above average = important
        is_small = ~is_large

        is_active = self.ternary != 0
        is_zero = ~is_active

        # ═══ Machine 1: PARTICIPATION (magnitude) ═══

        # For ZERO weights: large gradient = evidence to ACTIVATE
        #                   small gradient = evidence to STAY zero
        zero_delta = torch.where(is_large, self.c_mag_pos,
                                 torch.tensor(self.c_mag_neg, device=grad.device)).to(torch.int16)

        # For ACTIVE weights: small gradient = evidence to DEACTIVATE
        #                     large gradient = evidence to STAY active
        active_delta = torch.where(is_small, self.c_mag_pos,
                                   torch.tensor(self.c_mag_neg, device=grad.device)).to(torch.int16)

        # Apply: zeros accumulate activation evidence, actives accumulate deactivation evidence
        part_delta = torch.where(is_zero, zero_delta, active_delta)
        self.part_evidence.add_(part_delta)
        self.part_evidence.clamp_(min=0)
        # Decay for non-stationarity
        self.part_evidence.sub_(self.part_evidence >> 10)

        # ═══ Machine 2: SIGN (direction, active weights only) ═══

        grad_negative = grad < 0
        grad_positive = grad > 0

        up_delta = torch.where(grad_negative, self.c_sign_pos,
                    torch.where(grad_positive, self.c_sign_neg, 0)).to(torch.int16)
        down_delta = torch.where(grad_positive, self.c_sign_pos,
                      torch.where(grad_negative, self.c_sign_neg, 0)).to(torch.int16)

        self.sign_up.add_(up_delta)
        self.sign_down.add_(down_delta)
        self.sign_up.clamp_(min=0)
        self.sign_down.clamp_(min=0)
        self.sign_up.sub_(self.sign_up >> 10)
        self.sign_down.sub_(self.sign_down >> 10)

        # ═══ Transitions ═══

        # 1. Activation: zero → ±1 (participation evidence exceeds threshold)
        should_activate = is_zero & (self.part_evidence > self.tau_participation)
        # Pick sign from accumulated sign evidence
        activate_positive = should_activate & (self.sign_up >= self.sign_down)
        activate_negative = should_activate & (self.sign_down > self.sign_up)
        n_activate = should_activate.sum().item()

        # 2. Deactivation: ±1 → 0 (participation evidence exceeds threshold)
        should_deactivate = is_active & (self.part_evidence > self.tau_participation)
        n_deactivate = should_deactivate.sum().item()

        # 3. Sign flip: +1↔-1 (sign evidence exceeds threshold, active weights only)
        can_sign_flip_up = is_active & (self.ternary == -1) & (self.sign_up > self.tau_sign)
        can_sign_flip_down = is_active & (self.ternary == 1) & (self.sign_down > self.tau_sign)
        # Don't sign-flip if we're also deactivating
        can_sign_flip_up = can_sign_flip_up & ~should_deactivate
        can_sign_flip_down = can_sign_flip_down & ~should_deactivate
        n_sign = can_sign_flip_up.sum().item() + can_sign_flip_down.sum().item()

        # ═══ Apply ═══

        if n_activate > 0:
            self.ternary[activate_positive] = 1
            self.ternary[activate_negative] = -1

        if n_deactivate > 0:
            self.ternary[should_deactivate] = 0

        if n_sign > 0:
            self.ternary[can_sign_flip_up] = 1    # was -1 → +1
            self.ternary[can_sign_flip_down] = -1  # was +1 → -1

        # Reset evidence for all transitioned weights
        changed = should_activate | should_deactivate | can_sign_flip_up | can_sign_flip_down
        self.part_evidence[changed] = 0
        self.sign_up[changed] = 0
        self.sign_down[changed] = 0

        # ═══ Scale update ═══
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
        return {"n_activate": n_activate, "n_deactivate": n_deactivate, "n_sign": n_sign}


class DualSPRTEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 sprt: dict | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        if sprt is None:
            sprt = compute_dual_sprt_constants()
        self.c_sign_pos = sprt["c_sign_pos"]
        self.c_sign_neg = sprt["c_sign_neg"]
        self.tau_sign = sprt["tau_sign"]
        self.c_mag_pos = sprt["c_mag_pos"]
        self.c_mag_neg = sprt["c_mag_neg"]
        self.tau_participation = sprt["tau_participation"]

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = DualSPRTLinear._find_group_size(total, group_size)
        n_groups = total // self.group_size

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        self.register_buffer("part_evidence", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int16))
        self.register_buffer("sign_up", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int16))
        self.register_buffer("sign_down", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int16))
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
            return {"n_activate": 0, "n_deactivate": 0, "n_sign": 0}

        grad = self.weight.grad.detach()
        grad_abs = grad.abs()
        layer_mean = grad_abs.mean().clamp(min=1e-8)
        is_large = grad_abs > layer_mean
        is_small = ~is_large
        is_active = self.ternary != 0
        is_zero = ~is_active

        zero_delta = torch.where(is_large, self.c_mag_pos,
                                 torch.tensor(self.c_mag_neg, device=grad.device)).to(torch.int16)
        active_delta = torch.where(is_small, self.c_mag_pos,
                                   torch.tensor(self.c_mag_neg, device=grad.device)).to(torch.int16)
        self.part_evidence.add_(torch.where(is_zero, zero_delta, active_delta))
        self.part_evidence.clamp_(min=0).sub_(self.part_evidence >> 10)

        grad_negative = grad < 0
        grad_positive = grad > 0
        self.sign_up.add_(torch.where(grad_negative, self.c_sign_pos,
                           torch.where(grad_positive, self.c_sign_neg, 0)).to(torch.int16))
        self.sign_down.add_(torch.where(grad_positive, self.c_sign_pos,
                             torch.where(grad_negative, self.c_sign_neg, 0)).to(torch.int16))
        self.sign_up.clamp_(min=0).sub_(self.sign_up >> 10)
        self.sign_down.clamp_(min=0).sub_(self.sign_down >> 10)

        should_activate = is_zero & (self.part_evidence > self.tau_participation)
        activate_pos = should_activate & (self.sign_up >= self.sign_down)
        activate_neg = should_activate & (self.sign_down > self.sign_up)
        should_deactivate = is_active & (self.part_evidence > self.tau_participation)
        flip_up = is_active & (self.ternary == -1) & (self.sign_up > self.tau_sign) & ~should_deactivate
        flip_down = is_active & (self.ternary == 1) & (self.sign_down > self.tau_sign) & ~should_deactivate

        n_act = should_activate.sum().item()
        n_deact = should_deactivate.sum().item()
        n_sign = flip_up.sum().item() + flip_down.sum().item()

        if n_act > 0:
            self.ternary[activate_pos] = 1
            self.ternary[activate_neg] = -1
        if n_deact > 0:
            self.ternary[should_deactivate] = 0
        if n_sign > 0:
            self.ternary[flip_up] = 1
            self.ternary[flip_down] = -1

        changed = should_activate | should_deactivate | flip_up | flip_down
        self.part_evidence[changed] = 0
        self.sign_up[changed] = 0
        self.sign_down[changed] = 0

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
        return {"n_activate": n_act, "n_deactivate": n_deact, "n_sign": n_sign}


class DualSPRTAttention(nn.Module):
    def __init__(self, cfg, sprt):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5
        self.W_q = DualSPRTLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)
        self.W_k = DualSPRTLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)
        self.W_v = DualSPRTLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)
        self.W_o = DualSPRTLinear(cfg.d_model, cfg.d_model, cfg.group_size, sprt)
        if cfg.attn_residual:
            self.alpha = nn.Parameter(torch.full((cfg.n_heads, 1, 1), cfg.attn_residual_init))
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
            out = out + self.alpha * x.reshape(B, S, H, Dh).transpose(1, 2)
        return self.W_o(out.transpose(1, 2).reshape(B, S, D))


class DualSPRTMLP(nn.Module):
    def __init__(self, cfg, sprt):
        super().__init__()
        self.w_gate = DualSPRTLinear(cfg.d_model, cfg.d_ff, cfg.group_size, sprt)
        self.w_up = DualSPRTLinear(cfg.d_model, cfg.d_ff, cfg.group_size, sprt)
        self.w_down = DualSPRTLinear(cfg.d_ff, cfg.d_model, cfg.group_size, sprt)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DualSPRTBlock(nn.Module):
    def __init__(self, cfg, sprt):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = DualSPRTAttention(cfg, sprt)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = DualSPRTMLP(cfg, sprt)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DualSPRTConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.p_sign = kwargs.get("p_sign", 0.6)
        self.p_mag = kwargs.get("p_mag", 0.6)
        self.alpha_sign = kwargs.get("alpha_sign", 0.01)
        self.alpha_participation = kwargs.get("alpha_participation", 0.005)
        self.attn_residual = kwargs.get("attn_residual", True)
        self.attn_residual_init = kwargs.get("attn_residual_init", 0.1)


class DualSPRTTransformer(nn.Module):

    def __init__(self, cfg: DualSPRTConfig):
        super().__init__()
        self.cfg = cfg
        sprt = compute_dual_sprt_constants(cfg.p_sign, cfg.p_mag,
                                            cfg.alpha_sign, cfg.alpha_participation)

        self.embed = DualSPRTEmbedding(cfg.vocab_size, cfg.d_model, cfg.group_size, sprt)
        self.layers = nn.ModuleList([DualSPRTBlock(cfg, sprt) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = DualSPRTLinear(cfg.d_model, cfg.vocab_size, cfg.group_size, sprt)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_ternary = sum(m.ternary.numel() for m in self.modules() if hasattr(m, 'ternary'))
        opt_bytes = sum(m.part_evidence.numel()*2 + m.sign_up.numel()*2 + m.sign_down.numel()*2
                        for m in self.modules() if hasattr(m, 'part_evidence'))
        print(f"DualSPRTTransformer: {n_params/1e6:.1f}M params ({n_ternary/1e6:.1f}M ternary)")
        print(f"  Optimizer: {opt_bytes/1e6:.1f}MB ({opt_bytes/n_ternary:.1f} bytes/weight)")

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
        totals = {"n_activate": 0, "n_deactivate": 0, "n_sign": 0}
        for m in self.modules():
            if isinstance(m, (DualSPRTLinear, DualSPRTEmbedding)):
                s = m.flip_step(scale_lr=scale_lr)
                for k in totals:
                    totals[k] += s[k]
        totals["n_flips"] = totals["n_activate"] + totals["n_deactivate"] + totals["n_sign"]
        return totals

    def get_flip_params(self) -> list:
        return [m.weight for m in self.modules()
                if isinstance(m, (DualSPRTLinear, DualSPRTEmbedding))]

    def ternary_stats(self) -> dict:
        all_t = torch.cat([m.ternary.flatten().float()
                           for m in self.modules() if hasattr(m, 'ternary')])
        return {
            "neg_frac": (all_t == -1).float().mean().item(),
            "zero_frac": (all_t == 0).float().mean().item(),
            "pos_frac": (all_t == 1).float().mean().item(),
        }
