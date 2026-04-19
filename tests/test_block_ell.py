"""Tests for Block-ELL format and matmul."""

import jax
import jax.numpy as jnp
import pytest

from looped_blockell.layers.block_ell import (
    BlockELLConfig,
    BlockELLTensor,
    block_ell_matmul,
    block_ell_to_dense,
    create_block_ell_from_dense,
    create_random_topology,
)


def test_config_properties():
    cfg = BlockELLConfig(R=4, C=8, K=3, B=16)
    assert cfg.out_features == 64
    assert cfg.in_features == 128
    assert cfg.density == 0.375
    assert cfg.total_blocks == 12


def test_random_topology_shape():
    key = jax.random.PRNGKey(0)
    col_idx = create_random_topology(R=4, C=8, K=3, key=key)
    assert col_idx.shape == (4, 3)
    assert col_idx.dtype == jnp.int32
    for r in range(4):
        assert len(jnp.unique(col_idx[r])) == 3


def test_from_dense_roundtrip():
    key = jax.random.PRNGKey(42)
    dense = jax.random.normal(key, (64, 128))
    bell = create_block_ell_from_dense(dense, tile_size=16, density=1.0)
    recovered = block_ell_to_dense(bell)
    assert jnp.allclose(dense, recovered, atol=1e-5)


def test_matmul_matches_dense():
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    B_batch, S, d_in, d_out = 2, 8, 64, 32
    W = jax.random.normal(k1, (d_out, d_in))
    x = jax.random.normal(k2, (B_batch, S, d_in))

    bell = create_block_ell_from_dense(W, tile_size=16, density=1.0)
    y_sparse = block_ell_matmul(x, bell)
    y_dense = x @ W.T

    assert jnp.allclose(y_sparse, y_dense, atol=1e-4), \
        f"Max diff: {jnp.abs(y_sparse - y_dense).max()}"


def test_matmul_gradient_flow():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (32, 64))
    x = jax.random.normal(k2, (2, 4, 64))

    bell = create_block_ell_from_dense(W, tile_size=16, density=0.5)

    def loss_fn(values):
        bell_updated = BlockELLTensor(values=values, col_indices=bell.col_indices, config=bell.config)
        y = block_ell_matmul(x, bell_updated)
        return y.sum()

    grads = jax.grad(loss_fn)(bell.values)
    assert grads.shape == bell.values.shape
    assert jnp.any(grads != 0), "Gradients should be non-zero"


def test_matmul_sparse_density():
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (48, 96))
    x = jax.random.normal(k2, (1, 4, 96))

    bell_full = create_block_ell_from_dense(W, tile_size=16, density=1.0)
    bell_half = create_block_ell_from_dense(W, tile_size=16, density=0.5)

    y_full = block_ell_matmul(x, bell_full)
    y_half = block_ell_matmul(x, bell_half)

    assert y_full.shape == y_half.shape
    assert bell_half.config.K < bell_full.config.K
