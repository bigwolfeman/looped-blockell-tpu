"""MLP block (dense and block-sparse) for JAX/Flax TPU implementation."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from .block_ell import BlockELLConfig, BlockELLTensor, block_ell_matmul, create_random_topology


class MLPBlock(nn.Module):
    """Two-layer MLP with GELU.  Optionally uses block-sparse fc1/fc2.

    Dense path:
        x → fc1 (d_model → d_ff) → GELU → fc2 (d_ff → d_model)

    Block-sparse path:
        fc1 and fc2 weights are stored as block_ell values + col_indices in
        mutable 'block_ell' variable collection.  Topology is updated externally
        (prune/grow step).  Forward pass uses block_ell_matmul for both layers.

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension.
        dropout: Dropout probability (0 = no dropout).
        use_block_sparse: If True, use block-sparse matmul for fc1 and fc2.
        tile_size: Block size B; d_model and d_ff must be divisible.
        density: Initial fraction of active blocks per row.
        dtype: Compute dtype (default bfloat16).
    """

    d_model: int
    d_ff: int
    dropout: float = 0.0
    use_block_sparse: bool = False
    tile_size: int = 16
    density: float = 0.5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.use_block_sparse:
            return self._forward_sparse(x, deterministic)
        return self._forward_dense(x, deterministic)

    # ------------------------------------------------------------------
    # Dense path
    # ------------------------------------------------------------------

    def _forward_dense(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        hidden = nn.Dense(self.d_ff, use_bias=True, dtype=self.dtype, name="fc1")(x)
        hidden = jax.nn.gelu(hidden)
        out = nn.Dense(self.d_model, use_bias=True, dtype=self.dtype, name="fc2")(hidden)
        if self.dropout > 0.0:
            out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
        return out

    # ------------------------------------------------------------------
    # Block-sparse path
    # ------------------------------------------------------------------

    def _make_block_ell_config(self, out_features: int, in_features: int) -> BlockELLConfig:
        B = self.tile_size
        R = out_features // B
        C = in_features // B
        K = max(1, int(C * self.density))
        return BlockELLConfig(R=R, C=C, K=K, B=B)

    def _forward_sparse(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        B_tile = self.tile_size

        cfg1 = self._make_block_ell_config(self.d_ff, self.d_model)
        cfg2 = self._make_block_ell_config(self.d_model, self.d_ff)

        # fc1 block-ELL state — mutable so topology can be updated externally
        fc1_values = self.variable(
            "block_ell", "fc1_values",
            lambda: jnp.zeros((cfg1.R, cfg1.K, B_tile, B_tile), dtype=self.dtype),
        )
        fc1_cols = self.variable(
            "block_ell", "fc1_col_indices",
            lambda: create_random_topology(
                cfg1.R, cfg1.C, cfg1.K,
                jax.random.PRNGKey(0),
            ),
        )
        fc1_bias = self.variable(
            "block_ell", "fc1_bias",
            lambda: jnp.zeros((self.d_ff,), dtype=self.dtype),
        )

        # fc2 block-ELL state
        fc2_values = self.variable(
            "block_ell", "fc2_values",
            lambda: jnp.zeros((cfg2.R, cfg2.K, B_tile, B_tile), dtype=self.dtype),
        )
        fc2_cols = self.variable(
            "block_ell", "fc2_col_indices",
            lambda: create_random_topology(
                cfg2.R, cfg2.C, cfg2.K,
                jax.random.PRNGKey(1),
            ),
        )
        fc2_bias = self.variable(
            "block_ell", "fc2_bias",
            lambda: jnp.zeros((self.d_model,), dtype=self.dtype),
        )

        ell1 = BlockELLTensor(
            values=fc1_values.value,
            col_indices=fc1_cols.value,
            config=cfg1,
        )
        hidden = block_ell_matmul(x, ell1) + fc1_bias.value
        hidden = jax.nn.gelu(hidden)

        ell2 = BlockELLTensor(
            values=fc2_values.value,
            col_indices=fc2_cols.value,
            config=cfg2,
        )
        out = block_ell_matmul(hidden, ell2) + fc2_bias.value

        if self.dropout > 0.0:
            out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
        return out
