"""Pre-norm transformer block for JAX/Flax TPU implementation."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import flax.linen as nn

from .norms import RMSNorm
from .attention import MultiHeadAttention
from .mlp import MLPBlock


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with Block-ELL MLP.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    MLP always uses Block-ELL format (density=1.0 = dense-equivalent).
    """

    config: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        cfg = self.config

        norm1 = RMSNorm(eps=cfg.norm_eps, name="norm_attn")
        attn = MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            dtype=jnp.bfloat16,
            name="attention",
        )
        norm2 = RMSNorm(eps=cfg.norm_eps, name="norm_mlp")
        mlp = MLPBlock(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            tile_size=cfg.tile_size,
            density=cfg.initial_density,
            dtype=jnp.bfloat16,
            name="mlp",
        )

        x = x + attn(norm1(x), deterministic=deterministic)
        x = x + mlp(norm2(x), deterministic=deterministic)
        return x
