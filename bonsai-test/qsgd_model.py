"""QuantSGD: Ternary training as low-precision gradient descent.

Not combinatorial search. Not flip decisions. Just SGD on a quantized
representation. The error buffer IS the continuous weight state —
it never decays, it accumulates gradient updates, and the ternary
value is simply the rounded view.

effective_weight = ternary * scale + error
effective_weight -= lr * gradient
new_ternary = round(effective_weight / scale)
error = effective_weight - new_ternary * scale

Memory per weight:
  - ternary: int8 (2 bits effective)
  - error: fp16 (the sub-ternary optimization state)
  - momentum: fp16 (SGD momentum for smoothing)
  - scale: fp16 per group
  Total: ~5 bytes/param vs 10 bytes for shadow+Adam
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


class QSGDLinear(nn.Module):
    """Linear with quantized SGD — continuous optimization, discrete representation."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 momentum: float = 0.9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.momentum_decay = momentum

        total = out_features * in_features
        if total % group_size != 0:
            self.group_size = self._find_group_size(total, group_size)
        n_groups = total // self.group_size

        init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(init, a=5 ** 0.5)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(out_features, in_features).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))

        # The continuous state: error = effective_weight - ternary*scale
        # Seeded with quantization residual from init
        scales_exp = beta.expand(-1, self.group_size).reshape(out_features, in_features)
        residual = init - ternary.reshape(out_features, in_features).float() * scales_exp
        self.register_buffer("error", residual)

        # SGD momentum buffer
        self.register_buffer("grad_momentum", torch.zeros(out_features, in_features))

        # Scale optimizer (Adam on group scales)
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        # Gradient capture parameter
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    @staticmethod
    def _find_group_size(total: int, target: int) -> int:
        for gs in range(target, 0, -1):
            if total % gs == 0:
                return gs
        return 1

    def forward(self, x: Tensor) -> Tensor:
        gs = self.group_size
        n_groups = self.ternary.numel() // gs
        s = self.scales.unsqueeze(1).expand(n_groups, gs).reshape(self.out_features, self.in_features)
        with torch.no_grad():
            self.weight.data.copy_(self.ternary.float() * s)
        return F.linear(x, self.weight)

    def qsgd_step(self, lr: float, scale_lr: float = 1e-3) -> dict:
        """One step of quantized SGD. Just gradient descent + round."""
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        gs = self.group_size
        n_groups = self.ternary.numel() // gs

        # SGD with momentum on the effective weight
        self.grad_momentum.mul_(self.momentum_decay).add_(grad)

        # Update the continuous state: effective_weight -= lr * momentum
        self.error.sub_(self.grad_momentum, alpha=lr)

        # Reconstruct effective weight and re-round to ternary
        scale_exp = self.scales.unsqueeze(1).expand(n_groups, gs).reshape_as(self.ternary)
        effective = self.ternary.float() * scale_exp + self.error

        # Re-quantize: round(effective / scale) clamped to [-1, 1]
        normalized = effective / scale_exp.clamp(min=1e-8)
        new_ternary = normalized.round().clamp(-1, 1).to(torch.int8)

        # Track changes
        old_ternary = self.ternary
        changed = new_ternary != old_ternary
        was_zero = old_ternary == 0
        is_zero = new_ternary == 0
        n_sign = (changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (changed & (was_zero | is_zero)).sum().item()

        # Apply new ternary and update error to be the new residual
        self.ternary = new_ternary
        new_scale_exp = self.scales.unsqueeze(1).expand(n_groups, gs).reshape_as(self.ternary)
        self.error = effective - self.ternary.float() * new_scale_exp

        # Update group scales via Adam
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


class QSGDEmbedding(nn.Module):
    """Ternary embedding with quantized SGD."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 momentum: float = 0.9):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.momentum_decay = momentum

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = QSGDLinear._find_group_size(total, group_size)
        n_groups = total // self.group_size

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))

        scales_exp = beta.expand(-1, self.group_size).reshape(num_embeddings, embedding_dim)
        residual = init - ternary.reshape(num_embeddings, embedding_dim).float() * scales_exp
        self.register_buffer("error", residual)
        self.register_buffer("grad_momentum", torch.zeros(num_embeddings, embedding_dim))

        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))

    def forward(self, input_ids: Tensor) -> Tensor:
        gs = self.group_size
        n_groups = self.ternary.numel() // gs
        s = self.scales.unsqueeze(1).expand(n_groups, gs).reshape(self.num_embeddings, self.embedding_dim)
        with torch.no_grad():
            self.weight.data.copy_(self.ternary.float() * s)
        return F.embedding(input_ids, self.weight)

    def qsgd_step(self, lr: float, scale_lr: float = 1e-3) -> dict:
        if self.weight.grad is None:
            return {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        grad = self.weight.grad.detach()
        gs = self.group_size
        n_groups = self.ternary.numel() // gs

        self.grad_momentum.mul_(self.momentum_decay).add_(grad)
        self.error.sub_(self.grad_momentum, alpha=lr)

        scale_exp = self.scales.unsqueeze(1).expand(n_groups, gs).reshape_as(self.ternary)
        effective = self.ternary.float() * scale_exp + self.error

        normalized = effective / scale_exp.clamp(min=1e-8)
        new_ternary = normalized.round().clamp(-1, 1).to(torch.int8)

        old_ternary = self.ternary
        changed = new_ternary != old_ternary
        was_zero = old_ternary == 0
        is_zero = new_ternary == 0
        n_sign = (changed & ~was_zero & ~is_zero).sum().item()
        n_structural = (changed & (was_zero | is_zero)).sum().item()

        self.ternary = new_ternary
        self.error = effective - self.ternary.float() * scale_exp

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


class QSGDAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        gs, mom = cfg.group_size, cfg.momentum
        self.W_q = QSGDLinear(cfg.d_model, cfg.d_model, group_size=gs, momentum=mom)
        self.W_k = QSGDLinear(cfg.d_model, cfg.d_model, group_size=gs, momentum=mom)
        self.W_v = QSGDLinear(cfg.d_model, cfg.d_model, group_size=gs, momentum=mom)
        self.W_o = QSGDLinear(cfg.d_model, cfg.d_model, group_size=gs, momentum=mom)

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


class QSGDMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gs, mom = cfg.group_size, cfg.momentum
        self.w_gate = QSGDLinear(cfg.d_model, cfg.d_ff, group_size=gs, momentum=mom)
        self.w_up = QSGDLinear(cfg.d_model, cfg.d_ff, group_size=gs, momentum=mom)
        self.w_down = QSGDLinear(cfg.d_ff, cfg.d_model, group_size=gs, momentum=mom)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class QSGDBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = QSGDAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = QSGDMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class QSGDConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.momentum = kwargs.get("momentum", 0.9)


class QSGDTransformer(nn.Module):
    """Ternary transformer via quantized SGD.

    Not combinatorial search — just gradient descent where the
    weights happen to be rounded to ternary after each step.
    The error buffer is the sub-ternary continuous state.
    """

    def __init__(self, cfg: QSGDConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = QSGDEmbedding(cfg.vocab_size, cfg.d_model, group_size=cfg.group_size,
                                    momentum=cfg.momentum)
        self.layers = nn.ModuleList([QSGDBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = QSGDLinear(cfg.d_model, cfg.vocab_size, group_size=cfg.group_size,
                                   momentum=cfg.momentum)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_ternary = sum(m.ternary.numel() for m in self.modules() if hasattr(m, 'ternary'))
        print(f"QSGDTransformer: {n_params/1e6:.1f}M params "
              f"({n_ternary/1e6:.1f}M ternary, SGD+momentum, no shadow weights)")

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

    def flip_step(self, scale_lr: float = 1e-3, lr: float = 6e-4, **_kwargs) -> dict:
        """Quantized SGD step on all layers."""
        total = {"n_flips": 0, "n_sign": 0, "n_structural": 0}
        for m in self.modules():
            if isinstance(m, (QSGDLinear, QSGDEmbedding)):
                stats = m.qsgd_step(lr=lr, scale_lr=scale_lr)
                total["n_flips"] += stats["n_flips"]
                total["n_sign"] += stats["n_sign"]
                total["n_structural"] += stats["n_structural"]
        return total

    def get_flip_params(self) -> list:
        return [m.weight for m in self.modules()
                if isinstance(m, (QSGDLinear, QSGDEmbedding))]

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
