"""Tests for column reordering (d_ff permutation for macro-block density)."""

import jax
import jax.numpy as jnp
import pytest

from looped_blockell.opt.column_reorder import (
    compute_column_importance,
    compute_permutation,
    apply_permutation_to_block_ell,
    compute_k_active_macros,
    full_reorder_step,
)


def test_compute_column_importance_basic():
    R_model, K_fc1 = 4, 8
    C_ff = 12
    R_ff, K_fc2 = C_ff, 6

    fc1_alive = jnp.ones((R_model, K_fc1), dtype=jnp.bool_)
    fc1_cols = jax.random.randint(jax.random.PRNGKey(0), (R_model, K_fc1), 0, C_ff).astype(jnp.int32)
    fc2_alive = jnp.ones((R_ff, K_fc2), dtype=jnp.bool_)
    fc2_cols = jax.random.randint(jax.random.PRNGKey(1), (R_ff, K_fc2), 0, 4).astype(jnp.int32)

    importance = compute_column_importance(fc1_alive, fc2_alive, fc1_cols, fc2_cols, C_ff)
    assert importance.shape == (C_ff,)
    assert jnp.all(importance >= 0)


def test_compute_permutation_descending():
    importance = jnp.array([1.0, 5.0, 3.0, 0.0, 2.0])
    perm = compute_permutation(importance)
    assert perm[0] == 1  # highest importance first
    assert perm[-1] == 3  # lowest importance last


def test_apply_permutation_column_dim():
    R, K, B = 2, 4, 16
    C_ff = 6
    values = jnp.ones((R, K, B, B))
    col_indices = jnp.array([[0, 2, 4, 5], [1, 3, 4, 5]], dtype=jnp.int32)
    alive = jnp.ones((R, K), dtype=jnp.bool_)
    perm = jnp.array([5, 4, 3, 2, 1, 0], dtype=jnp.int32)  # reverse order

    new_vals, new_cols, new_alive = apply_permutation_to_block_ell(
        values, col_indices, alive, perm, is_column_dim=True
    )
    assert new_vals.shape == values.shape
    assert new_alive.shape == alive.shape
    # After reverse permutation: old col 0 -> new col 5, old col 2 -> new col 3, etc.
    inv_perm = jnp.argsort(perm)
    expected_row0 = inv_perm[col_indices[0]]
    assert jnp.array_equal(new_cols[0], expected_row0)


def test_apply_permutation_row_dim():
    R, K, B = 4, 3, 16
    values = jax.random.normal(jax.random.PRNGKey(0), (R, K, B, B))
    col_indices = jnp.arange(R * K, dtype=jnp.int32).reshape(R, K)
    alive = jnp.ones((R, K), dtype=jnp.bool_)
    perm = jnp.array([3, 1, 0, 2], dtype=jnp.int32)

    new_vals, new_cols, new_alive = apply_permutation_to_block_ell(
        values, col_indices, alive, perm, is_column_dim=False
    )
    assert jnp.array_equal(new_vals[0], values[3])
    assert jnp.array_equal(new_vals[1], values[1])


def test_compute_k_active_macros():
    R, K = 4, 8
    tiles_per_macro = 4
    C_ff = 16

    alive = jnp.ones((R, K), dtype=jnp.bool_)
    col_indices = jnp.tile(jnp.arange(K, dtype=jnp.int32), (R, 1))

    k_active = compute_k_active_macros(alive, col_indices, tiles_per_macro, C_ff)
    expected = int(jnp.ceil(K / tiles_per_macro))
    assert int(k_active) == expected


def test_k_active_macros_with_pruning():
    R, K = 2, 8
    tiles_per_macro = 4
    C_ff = 8

    # Only first 2 columns alive
    alive = jnp.array([[True, True, False, False, False, False, False, False],
                        [True, True, False, False, False, False, False, False]])
    col_indices = jnp.array([[0, 1, -1, -1, -1, -1, -1, -1],
                              [0, 1, -1, -1, -1, -1, -1, -1]], dtype=jnp.int32)

    k_active = compute_k_active_macros(alive, col_indices, tiles_per_macro, C_ff)
    assert int(k_active) == 1  # columns 0,1 fit in first macro-block


def test_dead_tiles_sentinel():
    R, K = 2, 4
    C_ff = 8
    values = jnp.ones((R, K, 16, 16))
    col_indices = jnp.array([[0, 2, -1, -1], [1, 3, -1, -1]], dtype=jnp.int32)
    alive = jnp.array([[True, True, False, False],
                        [True, True, False, False]])
    perm = jnp.arange(C_ff, dtype=jnp.int32)  # identity

    new_vals, new_cols, new_alive = apply_permutation_to_block_ell(
        values, col_indices, alive, perm, is_column_dim=True
    )
    # Dead tiles should keep sentinel -1
    assert jnp.all(new_cols[0, 2:] == -1)
    assert jnp.all(new_cols[1, 2:] == -1)
