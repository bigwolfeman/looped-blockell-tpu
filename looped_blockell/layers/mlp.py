"""Block-ELL MLP for JAX/Flax TPU implementation.

Always uses Block-ELL format. At density=1.0 (K=C), this is mathematically
equivalent to a dense matmul — XLA will optimize the gather when all tiles
are present.

Weight layout:
  values:      [R, K, B, B] — learnable params (in 'params', get gradients)
  col_indices: [R, K] int32 — topology (in 'topology' collection, updated by CMS)
  alive_mask:  [R, K] bool  — which tiles are active (in 'topology' collection)
  bias:        [out_features] — learnable params

Pruning: CMS zeros dead tiles' values, sets alive_mask=False.
Compaction: physically shrinks K, removes dead tile slots.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import flax.linen as nn

from .block_ell import BlockELLConfig, BlockELLTensor, block_ell_matmul, create_random_topology


class BlockELLLinear(nn.Module):
    """Single Block-ELL linear layer.

    values and bias are params (learnable). col_indices and alive_mask are
    topology state (mutable, updated externally by CMS prune/grow).
    """
    out_features: int
    in_features: int
    tile_size: int = 16
    density: float = 1.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        B = self.tile_size
        self.R = self.out_features // B
        self.C = self.in_features // B
        self.K = max(1, int(self.C * self.density))

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B = self.tile_size

        # Learnable weights — in 'params', get gradients
        values = self.param(
            "values",
            nn.initializers.normal(0.02),
            (self.R, self.K, B, B),
        )
        bias = self.param(
            "bias",
            nn.initializers.zeros,
            (self.out_features,),
        )

        # Topology state — in 'topology' collection, updated by CMS
        col_indices = self.variable(
            "topology", "col_indices",
            lambda: self._init_topology(),
        )
        alive_mask = self.variable(
            "topology", "alive_mask",
            lambda: jnp.ones((self.R, self.K), dtype=jnp.bool_),
        )

        # Zero dead tiles (alive_mask=False → values zeroed)
        masked_values = jnp.where(
            alive_mask.value[:, :, None, None],
            values.astype(self.dtype),
            jnp.zeros_like(values, dtype=self.dtype),
        )

        cfg = BlockELLConfig(R=self.R, C=self.C, K=self.K, B=B)
        ell = BlockELLTensor(values=masked_values, col_indices=col_indices.value, config=cfg)

        return block_ell_matmul(x, ell) + bias.astype(self.dtype)

    def _init_topology(self) -> jnp.ndarray:
        if self.density >= 1.0:
            # Full density: sequential column indices [0, 1, ..., C-1]
            return jnp.tile(
                jnp.arange(self.C, dtype=jnp.int32),
                (self.R, 1),
            )[:, :self.K]
        else:
            return create_random_topology(
                self.R, self.C, self.K,
                jax.random.PRNGKey(hash((self.out_features, self.in_features)) % 2**31),
            )


class MLPBlock(nn.Module):
    """Two-layer GELU MLP using Block-ELL sparse matmul.

    fc1: [d_model → d_ff], fc2: [d_ff → d_model].
    Both use BlockELLLinear — starts at density=1.0 (dense-equivalent),
    pruned by CMS during training, compacted before routing phase.
    """
    d_model: int
    d_ff: int
    dropout: float = 0.0
    tile_size: int = 16
    density: float = 1.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden = BlockELLLinear(
            out_features=self.d_ff,
            in_features=self.d_model,
            tile_size=self.tile_size,
            density=self.density,
            dtype=self.dtype,
            name="fc1",
        )(x)
        hidden = jax.nn.gelu(hidden)

        out = BlockELLLinear(
            out_features=self.d_model,
            in_features=self.d_ff,
            tile_size=self.tile_size,
            density=self.density,
            dtype=self.dtype,
            name="fc2",
        )(hidden)

        if self.dropout > 0.0:
            out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
        return out
