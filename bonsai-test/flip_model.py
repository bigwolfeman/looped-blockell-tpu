"""Flip-based ternary transformer — no shadow weights, no Adam on weights.

The Bonsai hypothesis: weights are ternary {-1, 0, +1}, updated via
discrete flip decisions driven by gradient-sign momentum. No continuous
weight storage. The bf16 gradient is ephemeral — computed and consumed
each step, then discarded.

Memory per weight parameter:
  - ternary value: int8 (could be packed to 1.58 bits)
  - flip momentum: fp16 (accumulated gradient sign signal)
  - group scale: fp16 (shared per 128 elements)
  Total: ~3.2 bytes vs 10 bytes for shadow+Adam = 3× less

Optimization is combinatorial search over ternary configurations,
guided by gradient sign accumulation. A weight only flips when
enough consecutive gradients agree on the direction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


class FlipLinear(nn.Module):
    """Linear layer with ternary weights updated by flip momentum.

    No shadow weights. No Adam for weights. The weight nn.Parameter exists
    ONLY for autograd gradient computation — it's overwritten each forward
    with ternary*scale, and its gradient is consumed in flip_step().
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128,
                 flip_threshold: float = 10.0, momentum_decay: float = 0.95):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.flip_threshold = flip_threshold
        self.momentum_decay = momentum_decay

        total = out_features * in_features
        if total % group_size != 0:
            self.group_size = self._find_group_size(total, group_size)
        n_groups = total // self.group_size

        # Initialize from kaiming → quantize to ternary
        init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(init, a=5 ** 0.5)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        # Persistent state: ternary values + scales + momentum
        self.register_buffer("ternary", ternary.reshape(out_features, in_features).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))  # [n_groups]
        self.register_buffer("flip_momentum", torch.zeros(out_features, in_features))

        # Scale optimizer state (tiny Adam just for group scales)
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        # Gradient capture: this parameter is overwritten each forward
        # and exists ONLY so autograd computes dL/dw
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    @staticmethod
    def _find_group_size(total: int, target: int) -> int:
        for gs in range(target, 0, -1):
            if total % gs == 0:
                return gs
        return 1

    def _reconstruct_weight(self) -> Tensor:
        """ternary * per-group scale → effective bf16 weight."""
        s = self.scales.unsqueeze(1).expand(-1, self.group_size)
        s = s.reshape(self.out_features, self.in_features)
        return self.ternary.float() * s

    def forward(self, x: Tensor) -> Tensor:
        # No input_norm or activation quantization here — block-level RMSNorm
        # handles normalization. Activation quantization is inference-only.
        with torch.no_grad():
            self.weight.data.copy_(self._reconstruct_weight())

        return F.linear(x, self.weight)

    def flip_step(self, scale_lr: float = 1e-3) -> int:
        """Consume gradient, update flip momentum, apply flips, update scales.

        Call after loss.backward(), before optimizer.step().
        Returns number of flips this step.
        """
        if self.weight.grad is None:
            return 0

        grad = self.weight.grad.detach()

        # --- 1. Update flip momentum ---
        # -sign(grad): negative gradient = loss decreases if weight increases
        self.flip_momentum.mul_(self.momentum_decay).add_(grad.sign(), alpha=-1.0)

        # --- 2. Identify flips ---
        want_up = self.flip_momentum > self.flip_threshold
        want_down = self.flip_momentum < -self.flip_threshold
        can_up = self.ternary < 1  # -1→0 or 0→+1
        can_down = self.ternary > -1  # +1→0 or 0→-1

        do_up = want_up & can_up
        do_down = want_down & can_down

        n_flips = do_up.sum().item() + do_down.sum().item()

        # --- 3. Apply flips ---
        if n_flips > 0:
            self.ternary[do_up] += 1
            self.ternary[do_down] -= 1

        # Reset momentum for flipped weights AND for boundary-saturated weights
        reset = do_up | do_down | (want_up & ~can_up) | (want_down & ~can_down)
        self.flip_momentum[reset] = 0.0

        # --- 4. Update group scales (tiny Adam) ---
        self.scale_step += 1
        gs = self.group_size
        # Per-group gradient magnitude: how much does loss want to change weight magnitude?
        grad_scale = (grad * self.ternary.float()).reshape(-1, gs).sum(dim=1)
        # This is dL/d(scale_g) = sum_i(ternary_i * dL/dw_i)

        b1, b2 = 0.9, 0.999
        self.scale_m1.lerp_(grad_scale, 1 - b1)
        self.scale_m2.lerp_(grad_scale.square(), 1 - b2)
        m1_hat = self.scale_m1 / (1 - b1 ** self.scale_step)
        m2_hat = self.scale_m2 / (1 - b2 ** self.scale_step)
        self.scales.addcdiv_(m1_hat, m2_hat.sqrt().add_(1e-8), value=-scale_lr)
        self.scales.clamp_(min=1e-6)

        # --- 5. Clear gradient (ephemeral — we consumed it) ---
        self.weight.grad = None

        return n_flips


class FlipEmbedding(nn.Module):
    """Ternary embedding with flip-based optimization.

    Each token's embedding is a ternary code: attract (+1), repel (-1), neutral (0).
    Flip momentum accumulates per-token-per-dim based on gradient signal.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128,
                 flip_threshold: float = 10.0, momentum_decay: float = 0.95):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.flip_threshold = flip_threshold
        self.momentum_decay = momentum_decay

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = FlipLinear._find_group_size(total, group_size)
        n_groups = total // self.group_size

        # Initialize
        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, std=0.02)
        beta = init.reshape(-1, self.group_size).abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        ternary = (init.reshape(-1, self.group_size) / beta).round().clamp(-1, 1)

        self.register_buffer("ternary", ternary.reshape(num_embeddings, embedding_dim).to(torch.int8))
        self.register_buffer("scales", beta.squeeze(1))
        self.register_buffer("flip_momentum", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("scale_m1", torch.zeros(n_groups))
        self.register_buffer("scale_m2", torch.zeros(n_groups))
        self.scale_step = 0

        # Gradient capture parameter
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

    def flip_step(self, scale_lr: float = 1e-3) -> int:
        if self.weight.grad is None:
            return 0

        grad = self.weight.grad.detach()
        self.flip_momentum.mul_(self.momentum_decay).add_(grad.sign(), alpha=-1.0)

        want_up = self.flip_momentum > self.flip_threshold
        want_down = self.flip_momentum < -self.flip_threshold
        do_up = want_up & (self.ternary < 1)
        do_down = want_down & (self.ternary > -1)

        n_flips = do_up.sum().item() + do_down.sum().item()
        if n_flips > 0:
            self.ternary[do_up] += 1
            self.ternary[do_down] -= 1

        reset = do_up | do_down | (want_up & (self.ternary >= 1)) | (want_down & (self.ternary <= -1))
        self.flip_momentum[reset] = 0.0

        # Scale update
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
        return n_flips


class FlipAttention(nn.Module):
    """Multi-head attention with FlipLinear projections + residual."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        gs = cfg.group_size
        ft = cfg.flip_threshold

        self.W_q = FlipLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_threshold=ft)
        self.W_k = FlipLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_threshold=ft)
        self.W_v = FlipLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_threshold=ft)
        self.W_o = FlipLinear(cfg.d_model, cfg.d_model, group_size=gs, flip_threshold=ft)

        if cfg.attn_residual:
            self.alpha = nn.Parameter(
                torch.full((cfg.n_heads, 1, 1), cfg.attn_residual_init)
            )
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


class FlipMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gs = cfg.group_size
        ft = cfg.flip_threshold
        self.w_gate = FlipLinear(cfg.d_model, cfg.d_ff, group_size=gs, flip_threshold=ft)
        self.w_up = FlipLinear(cfg.d_model, cfg.d_ff, group_size=gs, flip_threshold=ft)
        self.w_down = FlipLinear(cfg.d_ff, cfg.d_model, group_size=gs, flip_threshold=ft)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FlipBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = FlipAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = FlipMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class FlipTransformerConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.d_ff = kwargs.get("d_ff", 1376)
        self.n_layers = kwargs.get("n_layers", 6)
        self.vocab_size = kwargs.get("vocab_size", 49152)
        self.max_seq_len = kwargs.get("max_seq_len", 1024)
        self.group_size = kwargs.get("group_size", 128)
        self.flip_threshold = kwargs.get("flip_threshold", 10.0)
        self.momentum_decay = kwargs.get("momentum_decay", 0.95)
        self.attn_residual = kwargs.get("attn_residual", True)
        self.attn_residual_init = kwargs.get("attn_residual_init", 0.1)


class FlipTransformer(nn.Module):
    """Full-ternary transformer with flip-based discrete optimization.

    No shadow weights. No Adam on weight parameters.
    Training memory: ternary (int8) + momentum (fp16) + scales (fp16/128)
    """

    def __init__(self, cfg: FlipTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = FlipEmbedding(cfg.vocab_size, cfg.d_model, group_size=cfg.group_size,
                                    flip_threshold=cfg.flip_threshold)
        self.layers = nn.ModuleList([FlipBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = FlipLinear(cfg.d_model, cfg.vocab_size, group_size=cfg.group_size,
                                   flip_threshold=cfg.flip_threshold)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_ternary = sum(m.ternary.numel() for m in self.modules() if hasattr(m, 'ternary'))
        print(f"FlipTransformer: {n_params/1e6:.1f}M params "
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

    def flip_step(self, scale_lr: float = 1e-3) -> int:
        """Run flip updates on all FlipLinear/FlipEmbedding modules."""
        total_flips = 0
        for m in self.modules():
            if isinstance(m, (FlipLinear, FlipEmbedding)):
                total_flips += m.flip_step(scale_lr=scale_lr)
        return total_flips

    def get_flip_params(self) -> list:
        """Return weight parameters that should NOT be in the optimizer.
        These are the gradient-capture params managed by flip_step().
        """
        return [m.weight for m in self.modules()
                if isinstance(m, (FlipLinear, FlipEmbedding))]

    def ternary_stats(self) -> dict:
        """Return ternary weight distribution stats."""
        all_t = torch.cat([m.ternary.flatten().float()
                           for m in self.modules() if hasattr(m, 'ternary')])
        return {
            "neg_frac": (all_t == -1).float().mean().item(),
            "zero_frac": (all_t == 0).float().mean().item(),
            "pos_frac": (all_t == 1).float().mean().item(),
        }
