"""Block-ELL sparse tensor format in JAX.

Block-ELL (Block-ELLPACK) stores sparse matrices as:
  values: [R, K, B, B] — dense tile values
  col_indices: [R, K] — column-block index for each tile

Layout convention: weight matrix is [out_features, in_features] = [R*B, C*B].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass
class BlockELLConfig:
    R: int   # output block-rows
    C: int   # input block-columns
    K: int   # active blocks per row
    B: int = 16  # tile size

    def __post_init__(self) -> None:
        if self.K > self.C:
            raise ValueError(f"K ({self.K}) cannot exceed C ({self.C})")
        if any(v < 1 for v in (self.R, self.C, self.K, self.B)):
            raise ValueError("R, C, K, B must all be >= 1")

    @property
    def out_features(self) -> int:
        return self.R * self.B

    @property
    def in_features(self) -> int:
        return self.C * self.B

    @property
    def total_blocks(self) -> int:
        return self.R * self.K

    @property
    def total_parameters(self) -> int:
        return self.R * self.K * self.B * self.B

    @property
    def density(self) -> float:
        return self.K / self.C


@dataclass
class BlockELLTensor:
    values: jnp.ndarray      # [R, K, B, B]
    col_indices: jnp.ndarray  # [R, K] int32
    config: BlockELLConfig

    def __post_init__(self) -> None:
        cfg = self.config
        expected_v = (cfg.R, cfg.K, cfg.B, cfg.B)
        expected_i = (cfg.R, cfg.K)
        if self.values.shape != expected_v:
            raise ValueError(f"values shape {self.values.shape} != expected {expected_v}")
        if self.col_indices.shape != expected_i:
            raise ValueError(f"col_indices shape {self.col_indices.shape} != expected {expected_i}")


def create_random_topology(R: int, C: int, K: int, key: jax.Array) -> jnp.ndarray:
    """Return col_indices [R, K] with K unique columns per row, dtype int32."""
    keys = jax.random.split(key, R)

    def _sample_row(k):
        return jax.random.choice(k, C, shape=(K,), replace=False).astype(jnp.int32)

    return jax.vmap(_sample_row)(keys)  # [R, K]


def create_block_ell_from_dense(
    dense: jnp.ndarray,
    tile_size: int = 16,
    density: float = 0.5,
) -> BlockELLTensor:
    """Convert dense weight [out_features, in_features] → BlockELLTensor via top-K Frobenius norm."""
    out_features, in_features = dense.shape
    B = tile_size
    if out_features % B != 0 or in_features % B != 0:
        raise ValueError(
            f"Dimensions ({out_features}, {in_features}) must be divisible by tile_size {B}"
        )
    R = out_features // B
    C = in_features // B
    K = max(1, int(C * density))

    # Reshape into blocks: [R, C, B_in, B_out]
    # The einsum "bsrkd,rkdD->bsrD" contracts d (input sub-idx) so
    # values must be [R, K, B_in, B_out] (Flax [in, out] convention).
    blocks = dense.reshape(R, B, C, B).transpose(0, 2, 3, 1)  # [R, C, B_in, B_out]

    # Frobenius norm per block [R, C]
    block_norms = jnp.sqrt(jnp.sum(blocks ** 2, axis=(2, 3)))

    # Top-K per row — use argsort descending and take first K, then sort for consistency
    top_k_desc = jnp.argsort(-block_norms, axis=1)[:, :K]  # [R, K]
    col_indices = jnp.sort(top_k_desc, axis=1).astype(jnp.int32)  # [R, K] sorted

    # Gather selected block values
    # values[r, k] = blocks[r, col_indices[r, k]]
    def _gather_row(r_blocks, r_cols):
        return r_blocks[r_cols]  # [K, B, B]

    values = jax.vmap(_gather_row)(blocks, col_indices)  # [R, K, B, B]

    config = BlockELLConfig(R=R, C=C, K=K, B=B)
    return BlockELLTensor(values=values, col_indices=col_indices, config=config)


def block_ell_to_dense(block_ell: BlockELLTensor) -> jnp.ndarray:
    """Scatter BlockELLTensor → dense [out_features, in_features]."""
    cfg = block_ell.config
    R, K, B, C = cfg.R, cfg.K, cfg.B, cfg.C
    out_features = R * B
    in_features = C * B

    dense = jnp.zeros((out_features, in_features), dtype=block_ell.values.dtype)

    def _scatter_row(carry, inputs):
        d, r = carry, inputs
        r_values, r_cols = block_ell.values[r], block_ell.col_indices[r]

        def _scatter_block(d_inner, k):
            c = r_cols[k]
            row_s = r * B
            col_s = c * B
            # values are [B_in, B_out], dense is [out, in] → transpose
            block = r_values[k].T  # [B_in, B_out] → [B_out, B_in]
            d_inner = jax.lax.dynamic_update_slice(
                d_inner, block, (row_s, col_s)
            )
            return d_inner, None

        d, _ = jax.lax.scan(_scatter_block, d, jnp.arange(K))
        return d, None

    dense, _ = jax.lax.scan(_scatter_row, dense, jnp.arange(R))
    return dense


def block_ell_matmul(x: jnp.ndarray, block_ell: BlockELLTensor) -> jnp.ndarray:
    """Block-sparse matmul: x [B_batch, S, in_features] → [B_batch, S, out_features].

    Fully differentiable via JAX autograd.
    """
    cfg = block_ell.config
    R, K, B, C = cfg.R, cfg.K, cfg.B, cfg.C
    batch_size, seq_len, _ = x.shape

    # Reshape input into column tiles: [B_batch, S, C, B]
    x_tiled = x.reshape(batch_size, seq_len, C, B)

    # Gather input tiles for each active column: [B_batch, S, R, K, B]
    # col_indices: [R, K] → gather from x_tiled along axis 2
    # x_gathered[b, s, r, k, :] = x_tiled[b, s, col_indices[r, k], :]
    flat_cols = block_ell.col_indices.reshape(-1)                  # [R*K]
    x_flat = x_tiled[:, :, flat_cols, :]                           # [B_batch, S, R*K, B]
    x_gathered = x_flat.reshape(batch_size, seq_len, R, K, B)      # [B_batch, S, R, K, B]

    # Batched matmul: for each output block-row r, sum over K active blocks
    # values: [R, K, B, B]  (weight[r, k] maps in_B → out_B)
    # result[b, s, r, out_b] = sum_k  x_gathered[b, s, r, k, :] @ values[r, k].T
    # Use einsum: 'bsrkd, rkdD -> bsrD'
    output_blocked = jnp.einsum(
        "bsrkd,rkdD->bsrD",
        x_gathered,
        block_ell.values,
        precision=jax.lax.Precision.DEFAULT,
    )  # [B_batch, S, R, B]

    # Flatten output blocks: [B_batch, S, R*B] = [B_batch, S, out_features]
    out_features = R * B
    output = output_blocked.reshape(batch_size, seq_len, out_features)
    return output
