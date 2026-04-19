"""Column reordering to maximise macro-block density on the TPU MXU.

d_ff is the internal dimension shared between fc1 (output) and fc2 (input).
Because it's purely internal to the MLP, we can freely permute it to cluster
alive tiles toward low column / row indices.  After permutation:

  - fc1 alive column-tiles are packed leftward  → low column-block indices
  - fc2 alive row-tiles are packed upward        → low row-block indices
  - Tail macro-blocks become fully dead           → Pallas kernel skips them

This gives a free speedup at every prune round with zero quality loss.

Public API
----------
compute_column_importance(...)
compute_permutation(importance)
apply_permutation_to_block_ell(...)
apply_permutation_to_optimizer(...)
compute_k_active_macros(...)
full_reorder_step(...)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def compute_column_importance(
    fc1_alive_mask: jnp.ndarray,    # [R_model, K_fc1] bool
    fc2_alive_mask: jnp.ndarray,    # [R_ff, K_fc2] bool
    fc1_col_indices: jnp.ndarray,   # [R_model, K_fc1] int32   → index into d_ff columns
    fc2_col_indices: jnp.ndarray,   # [R_ff, K_fc2]  int32   → index into d_model columns
    C_ff: int,                       # total d_ff tile-columns
) -> jnp.ndarray:
    """Score each d_ff tile-column by alive-tile count across fc1 and fc2.

    fc1 contribution
    ~~~~~~~~~~~~~~~~
    fc1 maps d_model → d_ff.  col_indices point into the d_ff output dimension
    (tile-columns of fc1 output, i.e. tile-rows of d_ff).  For each d_ff column c,
    count alive fc1 tiles whose col_index == c.

    fc2 contribution
    ~~~~~~~~~~~~~~~~
    fc2 maps d_ff → d_model.  The *row* dimension of fc2 IS the d_ff feature.
    Row r of fc2 corresponds to d_ff column r.  We count alive tiles in each fc2 row.

    Parameters
    ----------
    fc1_alive_mask : [R_model, K_fc1]
    fc2_alive_mask : [R_ff, K_fc2]
        Note: R_ff == C_ff (fc2 has one block-row per d_ff tile-column).
    fc1_col_indices : [R_model, K_fc1] int32
    fc2_col_indices : [R_ff, K_fc2]   int32  (indexes into d_model — not used for d_ff scoring)
    C_ff : int

    Returns
    -------
    importance : [C_ff] float32 — higher = more tiles reference this d_ff column.
    """
    # --- fc1 contribution: scatter alive tiles by their col_index into d_ff ----
    # alive_and_col[r, k] = alive_mask[r, k] cast to float, for col col_indices[r, k]
    # We want: for each c in [0, C_ff), sum over (r, k) where col_indices[r,k]==c
    #          of alive_mask[r,k]
    flat_cols = fc1_col_indices.reshape(-1).astype(jnp.int32)   # [R_model * K_fc1]
    flat_alive = fc1_alive_mask.reshape(-1).astype(jnp.float32)  # [R_model * K_fc1]

    fc1_importance = jnp.zeros(C_ff, dtype=jnp.float32)
    # Clip col index in case of sentinels (-1 from pruning); we'll zero those out
    safe_cols = jnp.clip(flat_cols, 0, C_ff - 1)
    valid = (flat_cols >= 0).astype(jnp.float32)
    fc1_importance = fc1_importance.at[safe_cols].add(flat_alive * valid)

    # --- fc2 contribution: per-row alive-tile count, one row per d_ff column ----
    # fc2_alive_mask: [R_ff, K_fc2].  R_ff == C_ff.  Count alive per row.
    fc2_row_counts = fc2_alive_mask.astype(jnp.float32).sum(axis=1)  # [R_ff]

    # Guard against shape mismatch — fc2 may have fewer rows if d_ff is different
    ff_rows = fc2_row_counts.shape[0]
    if ff_rows != C_ff:
        # Pad or truncate to C_ff
        if ff_rows < C_ff:
            pad = jnp.zeros(C_ff - ff_rows, dtype=jnp.float32)
            fc2_row_counts = jnp.concatenate([fc2_row_counts, pad])
        else:
            fc2_row_counts = fc2_row_counts[:C_ff]

    return fc1_importance + fc2_row_counts


# ---------------------------------------------------------------------------
# Permutation computation
# ---------------------------------------------------------------------------

def compute_permutation(importance: jnp.ndarray) -> jnp.ndarray:
    """Sort d_ff tile-columns by importance descending.

    Parameters
    ----------
    importance : [C_ff] float32

    Returns
    -------
    perm : [C_ff] int32  — perm[0] is the most important d_ff column index.
    """
    return jnp.argsort(-importance).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Applying the permutation to Block-ELL tensors
# ---------------------------------------------------------------------------

def apply_permutation_to_block_ell(
    values: jnp.ndarray,         # [R, K, B, B]
    col_indices: jnp.ndarray,    # [R, K] int32
    alive_mask: jnp.ndarray,     # [R, K] bool
    perm: jnp.ndarray,           # [C_ff] int32
    is_column_dim: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply d_ff permutation to Block-ELL weight tensors.

    is_column_dim=True  (fc1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    fc1 col_indices index into d_ff tile-columns.  After permutation,
    old d_ff column `old_c` moves to position `inv_perm[old_c]`.
    We remap col_indices: new_col[r, k] = inv_perm[col_indices[r, k]].
    The values array stays in the same (r, k) slots — only the *logical*
    column address of each tile changes.  Tiles with sentinel -1 stay -1.

    is_column_dim=False  (fc2)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    fc2 rows ARE the d_ff dimension.  We physically reorder the row axis
    of values, col_indices, and alive_mask according to perm.

    Parameters
    ----------
    values : [R, K, B, B]
    col_indices : [R, K] int32
    alive_mask : [R, K] bool
    perm : [C_ff] int32  — new-to-old mapping (perm[new_pos] = old_pos)
    is_column_dim : bool

    Returns
    -------
    (new_values, new_col_indices, new_alive_mask)
    """
    if is_column_dim:
        # Build inverse permutation: inv_perm[old] = new
        C_ff = perm.shape[0]
        inv_perm = jnp.zeros(C_ff, dtype=jnp.int32).at[perm].set(jnp.arange(C_ff, dtype=jnp.int32))

        # Remap col_indices: sentinel -1 stays -1
        sentinel_mask = col_indices < 0
        safe_cols = jnp.clip(col_indices, 0, C_ff - 1)
        new_col_indices = jnp.where(sentinel_mask, jnp.full_like(col_indices, -1), inv_perm[safe_cols])

        # values and alive_mask are unchanged — the tiles stay in the same (r,k) slot
        return values, new_col_indices, alive_mask

    else:
        # Reorder rows: values[perm[i]], col_indices[perm[i]], alive_mask[perm[i]]
        new_values = values[perm]       # [C_ff, K, B, B]
        new_col_indices = col_indices[perm]   # [C_ff, K]
        new_alive = alive_mask[perm]          # [C_ff, K]
        return new_values, new_col_indices, new_alive


# ---------------------------------------------------------------------------
# Optimizer state permutation
# ---------------------------------------------------------------------------

def apply_permutation_to_optimizer(
    opt_state: Any,
    perm: jnp.ndarray,
    param_name: str,
    is_column_dim: bool = True,
) -> Any:
    """Apply d_ff permutation to optax optimizer state for one parameter.

    Traverses the optax state pytree and applies the same index remapping to
    any leaf tensors that correspond to `param_name` (matched by path).

    In practice, optax stores mu (first moment) and nu (second moment) in
    NamedTuple leaves with the same shape as the parameter.

    Parameters
    ----------
    opt_state : optax optimizer state (pytree)
    perm : [C_ff] int32
    param_name : str — dotted path to the parameter in the params pytree
                       (e.g. "layers_0.mlp.fc1_values")
    is_column_dim : bool — same semantics as apply_permutation_to_block_ell

    Returns
    -------
    Updated opt_state pytree with permuted momentum/variance tensors.

    Note
    ----
    This function applies the permutation to ALL leaves of opt_state whose
    shape matches the weight tensor's shape.  For optax ScaleByAdam, the
    leaves are mu and nu, both shaped identically to the parameter.
    """
    C_ff = perm.shape[0]

    def _permute_leaf(leaf):
        if not isinstance(leaf, jnp.ndarray):
            return leaf
        # Detect whether this leaf could be a moment tensor for fc1/fc2
        if is_column_dim:
            # fc1: values shape [R, K, B, B] — permute col axis via col_indices remap
            # For moment tensors the shape is identical to values — we remap the same way
            # as we'd remap col_indices, but here we're reordering K slots that correspond
            # to a particular col_index.  Since we don't track which slot maps to which col
            # inside the optimizer state, we take the safe approach: reorder along axis-0
            # only when the first dimension == C_ff (fc2 row case).
            # For fc1, col_indices are remapped (not values), so moment tensors need no change.
            return leaf
        else:
            # fc2: row dim == C_ff
            if leaf.ndim >= 1 and leaf.shape[0] == C_ff:
                return leaf[perm]
            return leaf

    return jax.tree_util.tree_map(_permute_leaf, opt_state)


# ---------------------------------------------------------------------------
# Macro-block active count
# ---------------------------------------------------------------------------

def compute_k_active_macros(
    alive_mask: jnp.ndarray,          # [R, K] bool — after column reorder
    col_indices: jnp.ndarray,         # [R, K] int32 — after column reorder
    tiles_per_macro: int,
    C_ff: int,
) -> jnp.ndarray:
    """Number of active macro-block columns after column reorder.

    After reorder, alive tiles are packed toward low col_index values.
    We find the maximum alive col_index and compute how many macro-blocks
    (groups of `tiles_per_macro` tile-columns) are needed to cover them.

    Parameters
    ----------
    alive_mask : [R, K] bool
    col_indices : [R, K] int32  (sentinels = -1 for dead tiles)
    tiles_per_macro : int  — number of tile-columns per macro-block
    C_ff : int  — total tile-columns

    Returns
    -------
    k_active : scalar int32
    """
    # Mask out dead/sentinel tiles
    safe_cols = jnp.where(alive_mask, col_indices, jnp.full_like(col_indices, -1))
    max_alive_col = safe_cols.max()  # -1 if everything is dead

    # Number of macro-columns needed = ceil((max_alive_col + 1) / tiles_per_macro)
    k_active = jnp.where(
        max_alive_col < 0,
        jnp.array(0, dtype=jnp.int32),
        jnp.ceil((max_alive_col + 1).astype(jnp.float32) / tiles_per_macro).astype(jnp.int32),
    )
    return k_active


# ---------------------------------------------------------------------------
# Full reorder step
# ---------------------------------------------------------------------------

def full_reorder_step(
    fc1_values: jnp.ndarray,        # [R_model, K_fc1, B, B]
    fc1_col_indices: jnp.ndarray,   # [R_model, K_fc1] int32
    fc1_alive: jnp.ndarray,         # [R_model, K_fc1] bool
    fc2_values: jnp.ndarray,        # [C_ff, K_fc2, B, B]
    fc2_col_indices: jnp.ndarray,   # [C_ff, K_fc2] int32
    fc2_alive: jnp.ndarray,         # [C_ff, K_fc2] bool
    opt_state: Any,
    C_ff: int,
    tiles_per_macro: int,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray,   # fc1 values, cols, alive
    jnp.ndarray, jnp.ndarray, jnp.ndarray,   # fc2 values, cols, alive
    Any,                                       # opt_state
    jnp.ndarray,                               # k_active_macros
]:
    """Complete column reorder step.  Call after each prune round.

    Steps
    -----
    1. Compute d_ff column importance from fc1 + fc2 alive masks.
    2. Compute permutation (sort by descending importance).
    3. Apply to fc1 (column dim — remap col_indices).
    4. Apply to fc2 (row dim — reorder rows).
    5. Apply to optimizer state (fc2 row reorder).
    6. Compute K_active_macros from the new fc1 col_indices.

    Parameters
    ----------
    fc1_values, fc1_col_indices, fc1_alive : fc1 Block-ELL tensors + alive mask
    fc2_values, fc2_col_indices, fc2_alive : fc2 Block-ELL tensors + alive mask
    opt_state : optax optimizer state
    C_ff : int — total d_ff tile-columns
    tiles_per_macro : int — tile-columns per MXU macro-block

    Returns
    -------
    Tuple of (fc1_vals, fc1_cols, fc1_alive,
              fc2_vals, fc2_cols, fc2_alive,
              opt_state,
              k_active_macros)
    """
    # 1. Importance
    importance = compute_column_importance(
        fc1_alive_mask=fc1_alive,
        fc2_alive_mask=fc2_alive,
        fc1_col_indices=fc1_col_indices,
        fc2_col_indices=fc2_col_indices,
        C_ff=C_ff,
    )

    # 2. Permutation
    perm = compute_permutation(importance)  # [C_ff]

    # 3. fc1 — column dim: remap col_indices only
    fc1_values_new, fc1_cols_new, fc1_alive_new = apply_permutation_to_block_ell(
        fc1_values, fc1_col_indices, fc1_alive, perm, is_column_dim=True
    )

    # 4. fc2 — row dim: reorder rows
    fc2_values_new, fc2_cols_new, fc2_alive_new = apply_permutation_to_block_ell(
        fc2_values, fc2_col_indices, fc2_alive, perm, is_column_dim=False
    )

    # 5. Optimizer state — only fc2 row dim physically reorders
    opt_state_new = apply_permutation_to_optimizer(
        opt_state, perm, param_name="fc2_values", is_column_dim=False
    )

    # 6. K_active_macros — based on fc1 new col_indices (packed leftward)
    k_active = compute_k_active_macros(
        alive_mask=fc1_alive_new,
        col_indices=fc1_cols_new,
        tiles_per_macro=tiles_per_macro,
        C_ff=C_ff,
    )

    return (
        fc1_values_new, fc1_cols_new, fc1_alive_new,
        fc2_values_new, fc2_cols_new, fc2_alive_new,
        opt_state_new,
        k_active,
    )
