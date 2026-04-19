"""Multi-head causal attention with RoPE for JAX/Flax TPU implementation."""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


def _precompute_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> jnp.ndarray:
    """Precompute RoPE frequency table [max_seq_len, head_dim // 2, 2].

    Returns complex-valued rotation factors as real pairs [cos, sin].
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    # t: [T], inv_freq: [half] → freqs: [T, half]
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # [T, half]
    return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)  # [T, half, 2]


def _apply_rope(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary position embedding to x [B, n_heads, S, head_dim].

    freqs: [S, head_dim // 2, 2]  (cos, sin pairs)
    """
    B, H, S, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]          # each [B, H, S, half]
    cos = freqs[:S, :, 0]                            # [S, half]
    sin = freqs[:S, :, 1]                            # [S, half]
    # Broadcast over batch and head dims
    cos = cos[None, None, :, :]                      # [1, 1, S, half]
    sin = sin[None, None, :, :]
    rotated = jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
    )
    return rotated


class MultiHeadAttention(nn.Module):
    n_heads: int
    d_model: int
    max_seq_len: int = 1024
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: [B, S, d_model]
            deterministic: If False, apply dropout.

        Returns:
            [B, S, d_model]
        """
        B, S, D = x.shape
        head_dim = self.d_model // self.n_heads
        assert D == self.d_model

        # Q / K / V projections — single fused linear then split
        qkv = nn.Dense(
            3 * self.d_model,
            use_bias=False,
            dtype=self.dtype,
            name="qkv_proj",
        )(x)  # [B, S, 3*D]

        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [B, S, D]

        def _reshape(t):
            return t.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)  # [B, H, S, head_dim]

        # RoPE — precompute once as a constant (no params, just math)
        freqs = _precompute_freqs(head_dim, self.max_seq_len)
        freqs = freqs.astype(jnp.float32)  # keep freqs in fp32 for precision
        q = _apply_rope(q.astype(jnp.float32), freqs).astype(self.dtype)
        k = _apply_rope(k.astype(jnp.float32), freqs).astype(self.dtype)

        # Causal mask — [1, 1, S, S]
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))[None, None, :, :]
        additive_mask = jnp.where(mask, jnp.zeros((S, S), dtype=self.dtype), jnp.finfo(self.dtype).min)

        # Scaled dot-product attention
        scale = math.sqrt(head_dim)
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # [B, H, S, S]
        attn_logits = attn_logits + additive_mask
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(self.dtype)

        if self.dropout > 0.0:
            attn_weights = nn.Dropout(rate=self.dropout)(attn_weights, deterministic=deterministic)

        attn_out = jnp.matmul(attn_weights, v)  # [B, H, S, head_dim]

        # Merge heads
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)

        # Output projection
        out = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype,
            name="out_proj",
        )(attn_out)

        return out
