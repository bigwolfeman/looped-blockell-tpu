"""Attention optimizations: GQA, QK-Norm, XSA, Residual, CoPE, combined.

Each optimization is registered as its own attention type for independent ablation.
All inherit from BaseAttention and modify the dense attention formula.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from attention import BaseAttention, RotaryEmbedding, register_attention, _rotate_half


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ─── GQA: Grouped Query Attention ───────────────────────────────────────────

@register_attention("gqa")
class GQAAttention(BaseAttention):
    """Dense attention with grouped KV heads (n_kv_heads < n_heads).

    Each KV head serves n_heads/n_kv_heads query heads via repeat_interleave.
    With n_heads=12, n_kv_heads=4: 3:1 ratio, 66% KV param reduction.
    """

    def __init__(self, cfg, layer_idx: int = 0, n_kv_heads: int = 4, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        assert cfg.n_heads % n_kv_heads == 0, (
            f"n_heads={cfg.n_heads} must be divisible by n_kv_heads={n_kv_heads}")
        self.n_kv_heads = n_kv_heads
        self.n_rep = cfg.n_heads // n_kv_heads
        self.W_k = nn.Linear(cfg.d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(cfg.d_model, n_kv_heads * self.d_head, bias=False)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, S, _ = x.shape
        q = self.W_q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)
        return self.W_o(out)


# ─── QK-Norm: Gemma 2 / Chameleon Style ─────────────────────────────────────

@register_attention("qk_norm")
class QKNormAttention(BaseAttention):
    """Dense attention with RMSNorm on Q and K after projection, before RoPE.

    Prevents attention logit growth at scale. The norm is applied before RoPE
    so positional encoding magnitude is preserved.
    """

    def __init__(self, cfg, layer_idx: int = 0, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        self.q_norm = _RMSNorm(self.d_head)
        self.k_norm = _RMSNorm(self.d_head)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, S, _ = x.shape
        q = self.W_q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)
        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)
        return self.W_o(out)


# ─── XSA: Exclusive Self Attention ───────────────────────────────────────────

@register_attention("xsa")
class XSAAttention(BaseAttention):
    """Dense attention excluding self-token (diagonal masked out).

    Each query attends to all causal positions EXCEPT its own.
    Prevents identity shortcut (token attending primarily to itself).
    First token has no valid keys → gets zero attention output.
    """

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        B, H, S, D = q.shape
        mask = torch.tril(torch.ones(S, S, device=x.device))
        mask = mask - torch.eye(S, device=x.device)
        attn_mask = torch.where(mask.bool(), 0.0, float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask.unsqueeze(0).unsqueeze(0), scale=D ** -0.5
        )
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)


# ─── Residual Attention ──────────────────────────────────────────────────────

@register_attention("residual_attn")
class ResidualAttention(BaseAttention):
    """Dense attention with learned residual: out = W_o(attn + α·x_heads).

    α is a learned scalar per head, initialized to 0.1.
    Guarantees gradient flow even if attention collapses.
    Proven effective in our ternary model work.
    """

    def __init__(self, cfg, layer_idx: int = 0, init_alpha: float = 0.1, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        self.alpha = nn.Parameter(torch.full((cfg.n_heads, 1, 1), init_alpha))

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        B, H, S, D = q.shape
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x_heads = x.reshape(B, S, H, D).transpose(1, 2)
        out = attn_out + self.alpha * x_heads
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)


# ─── CoPE: Clipped RoPE (arXiv 2602.05258) ──────────────────────────────────

class CoPEEmbedding(RotaryEmbedding):
    """Clipped RoPE: cosine-taper low-frequency components.

    Frequencies with wavelength > context_len are gradually attenuated.
    Multiplying inv_freq by taper shrinks the rotation angle toward 0,
    giving cos→1, sin→0 (identity) for fully clipped dimensions.
    """

    def __init__(self, d_head: int, max_seq_len: int = 32768,
                 base: float = 10000.0, context_len: int = 4096):
        nn.Module.__init__(self)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))

        wavelengths = 2 * math.pi / inv_freq
        taper = torch.ones_like(inv_freq)
        long_mask = wavelengths > context_len
        if long_mask.any():
            log_w = torch.log(wavelengths[long_mask])
            log_L = math.log(context_len)
            log_max = torch.log(wavelengths).max()
            ratio = (log_w - log_L) / (log_max - log_L + 1e-8)
            taper[long_mask] = torch.cos(ratio * math.pi / 2).clamp(min=0)

        tapered_inv_freq = inv_freq * taper
        self.register_buffer("inv_freq", tapered_inv_freq, persistent=False)
        self._build_cache(max_seq_len)


@register_attention("dense_cope")
class DenseCoPEAttention(BaseAttention):
    """Dense attention with CoPE (Clipped RoPE) for length extrapolation.

    Plug-and-play: only modifies frequency weights at init.
    +10.84% HELMET avg within training range, ~2x at 256k (paper results).
    Zero inference overhead.
    """

    def __init__(self, cfg, layer_idx: int = 0, context_len: int = 4096, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        self.rope = CoPEEmbedding(
            self.d_head, cfg.max_seq_len, context_len=context_len
        )

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)
        return self.W_o(out)


# ─── Combined: Dense Optimized ───────────────────────────────────────────────

@register_attention("dense_opt")
class DenseOptimizedAttention(BaseAttention):
    """Combined: GQA + QK-Norm + Residual + CoPE."""

    def __init__(self, cfg, layer_idx: int = 0,
                 n_kv_heads: int = 4, context_len: int = 4096,
                 init_alpha: float = 0.1, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)

        assert cfg.n_heads % n_kv_heads == 0
        self.n_kv_heads = n_kv_heads
        self.n_rep = cfg.n_heads // n_kv_heads
        self.W_k = nn.Linear(cfg.d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(cfg.d_model, n_kv_heads * self.d_head, bias=False)

        self.q_norm = _RMSNorm(self.d_head)
        self.k_norm = _RMSNorm(self.d_head)
        self.alpha = nn.Parameter(torch.full((cfg.n_heads, 1, 1), init_alpha))
        self.rope = CoPEEmbedding(
            self.d_head, cfg.max_seq_len, context_len=context_len
        )

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, S, _ = x.shape
        q = self.W_q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        B, H, S, D = q.shape
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x_heads = x.reshape(B, S, H, D).transpose(1, 2)
        out = attn_out + self.alpha * x_heads
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)
