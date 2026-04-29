"""Pre-norm transformer block for JAX/Flax TPU implementation."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import flax.linen as nn

from .norms import RMSNorm
from .attention import MultiHeadAttention
from .sparse_attention import DeepSeekSparseAttention
from .compressed_sparse_attention import CompressedSparseAttention
from .mlp import MLPBlock


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with Block-ELL MLP.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    MLP always uses Block-ELL format (density=1.0 = dense-equivalent).
    Attention uses standard causal or DeepSeek sparse depending on config.
    """

    config: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        cfg = self.config

        norm1 = RMSNorm(eps=cfg.norm_eps, name="norm_attn")

        if cfg.use_sparse_attention and cfg.sparse_attn_type == "csa":
            attn = CompressedSparseAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                max_seq_len=cfg.max_seq_len,
                compress_ratio=cfg.csa_compress_ratio,
                compress_stride=cfg.csa_compress_stride,
                top_k=cfg.sparse_attn_top_k,
                window_size=cfg.csa_window_size,
                n_indexer_heads=cfg.sparse_attn_n_indexer_heads,
                name="attention",
            )
        elif cfg.use_sparse_attention:
            attn = DeepSeekSparseAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                max_seq_len=cfg.max_seq_len,
                top_k=cfg.sparse_attn_top_k,
                n_indexer_heads=cfg.sparse_attn_n_indexer_heads,
                block_size=cfg.sparse_attn_block_size,
                name="attention",
            )
        else:
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
