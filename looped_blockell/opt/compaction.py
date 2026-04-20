"""Physical compaction: shrink Block-ELL K, column reorder, rebuild optimizer.

Called once at prune_end. Takes Block-ELL with dead tiles (alive_mask has Falses)
and physically removes dead slots, producing a smaller K_new < K_old.

Steps:
1. For each Block-ELL layer: pack alive tiles to front, truncate K to K_new
2. Column reorder d_ff for macro-block density
3. Rebuild params pytree with new shapes
4. Rebuild optimizer state for new shapes
5. Add router params + iteration embedding params
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax

from .column_reorder import (
    compute_column_importance,
    compute_permutation,
    apply_permutation_to_block_ell,
    compute_k_active_macros,
)


def compact_single_layer(
    values: jnp.ndarray,      # [R, K_old, B, B]
    col_indices: jnp.ndarray,  # [R, K_old]
    alive_mask: jnp.ndarray,   # [R, K_old]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Pack alive tiles to front, compute K_new.

    Returns (new_values, new_col_indices, new_alive_mask, K_new).
    All have shape [R, K_new, ...] where K_new = max alive per row.
    """
    R, K_old, B, _ = values.shape

    alive_per_row = alive_mask.sum(axis=1)  # [R]
    K_new = int(alive_per_row.max())
    K_new = max(K_new, 1)

    def _pack_row(args):
        vals_row, cols_row, alive_row = args  # [K_old, B, B], [K_old], [K_old]

        # Sort: alive tiles first (alive=True sorts before False when negated)
        order = jnp.argsort(~alive_row)  # alive first
        vals_sorted = vals_row[order]
        cols_sorted = cols_row[order]
        alive_sorted = alive_row[order]

        # Truncate to K_new
        return vals_sorted[:K_new], cols_sorted[:K_new], alive_sorted[:K_new]

    new_vals, new_cols, new_alive = jax.vmap(_pack_row)((values, col_indices, alive_mask))
    return new_vals, new_cols, new_alive, K_new


def compact_params_and_topology(
    params: Any,
    topology: Any,
    bell_paths: List[Tuple[str, str, str]],
) -> Tuple[Any, Any, Dict[str, int]]:
    """Compact all Block-ELL layers: shrink K, remove dead slots.

    Args:
        params: Full params pytree (values are under dotted paths)
        topology: Topology pytree (col_indices, alive_mask)
        bell_paths: List of (values_path, layer_name, fc_name)

    Returns:
        (new_params, new_topology, k_new_map)
        k_new_map: {path: K_new} for each compacted layer
    """
    from . import _set_nested, _get_nested  # will define below

    k_new_map = {}

    for path, layer, fc in bell_paths:
        values = _get_nested(params, path)
        topo_prefix = f"{layer}.mlp.{fc}"
        col_indices = _get_nested(topology, f"{topo_prefix}.col_indices")
        alive_mask = _get_nested(topology, f"{topo_prefix}.alive_mask")

        R, K_old = values.shape[0], values.shape[1]
        old_alive = int(alive_mask.sum())

        new_vals, new_cols, new_alive, K_new = compact_single_layer(
            values, col_indices, alive_mask
        )

        params = _set_nested(params, path, new_vals)
        topology = _set_nested(topology, f"{topo_prefix}.col_indices", new_cols)
        topology = _set_nested(topology, f"{topo_prefix}.alive_mask", new_alive)

        k_new_map[path] = K_new
        print(f"  {path}: K {K_old} → {K_new} ({old_alive} alive tiles packed)")

    return params, topology, k_new_map


def column_reorder_all(
    params: Any,
    topology: Any,
    bell_paths: List[Tuple[str, str, str]],
    n_core: int,
    macro_tile_size: int = 128,
    tile_size: int = 16,
) -> Tuple[Any, Any, Dict[str, int]]:
    """Column reorder d_ff for each core layer's fc1+fc2 pair.

    Returns (params, topology, k_active_macros).
    """
    tiles_per_macro = macro_tile_size // tile_size
    k_active_macros = {}

    # Group bell_paths by layer
    layer_fcs = {}
    for path, layer, fc in bell_paths:
        if "core_" not in layer:
            continue
        layer_fcs.setdefault(layer, {})[fc] = path

    for layer, fc_paths in layer_fcs.items():
        if "fc1" not in fc_paths or "fc2" not in fc_paths:
            continue

        fc1_path = fc_paths["fc1"]
        fc2_path = fc_paths["fc2"]
        topo1 = f"{layer}.mlp.fc1"
        topo2 = f"{layer}.mlp.fc2"

        fc1_vals = _get_nested(params, fc1_path)
        fc1_cols = _get_nested(topology, f"{topo1}.col_indices")
        fc1_alive = _get_nested(topology, f"{topo1}.alive_mask")
        fc2_vals = _get_nested(params, fc2_path)
        fc2_cols = _get_nested(topology, f"{topo2}.col_indices")
        fc2_alive = _get_nested(topology, f"{topo2}.alive_mask")

        C_ff = fc1_vals.shape[0]  # R for fc1 = d_ff/B... wait
        # fc1: out=d_ff, in=d_model → R=d_ff/B, col_indices index into d_model columns
        # fc2: out=d_model, in=d_ff → R=d_model/B, col_indices index into d_ff columns
        # d_ff dimension: fc1 rows, fc2 col_indices
        # Hmm, for column reorder of d_ff:
        # - fc1: d_ff is the OUTPUT dim (row blocks). Permuting d_ff = permuting fc1 rows.
        # - fc2: d_ff is the INPUT dim (col_indices). Permuting d_ff = remapping fc2 col_indices.

        R_fc1 = fc1_vals.shape[0]  # d_ff / B
        C_ff_tiles = R_fc1  # number of d_ff tile-columns = R of fc1

        # Args are intentionally swapped: compute_column_importance's "fc1" args
        # should be the layer whose col_indices point INTO d_ff (= fc2, since
        # fc2 has in=d_ff). Its "fc2" args should be the layer whose ROWS are
        # d_ff features (= fc1, since fc1 has out=d_ff → R=d_ff/B rows).
        importance = compute_column_importance(
            fc1_alive_mask=fc2_alive,
            fc2_alive_mask=fc1_alive,
            fc1_col_indices=fc2_cols,
            fc2_col_indices=fc1_cols,
            C_ff=C_ff_tiles,
        )
        perm = compute_permutation(importance)

        # fc1: rows ARE d_ff → reorder rows (is_column_dim=False)
        fc1_v, fc1_c, fc1_a = apply_permutation_to_block_ell(
            fc1_vals, fc1_cols, fc1_alive, perm, is_column_dim=False
        )
        params = _set_nested(params, fc1_path, fc1_v)
        topology = _set_nested(topology, f"{topo1}.col_indices", fc1_c)
        topology = _set_nested(topology, f"{topo1}.alive_mask", fc1_a)

        # fc2: col_indices reference d_ff → remap col_indices (is_column_dim=True)
        fc2_v, fc2_c, fc2_a = apply_permutation_to_block_ell(
            fc2_vals, fc2_cols, fc2_alive, perm, is_column_dim=True
        )
        params = _set_nested(params, fc2_path, fc2_v)
        topology = _set_nested(topology, f"{topo2}.col_indices", fc2_c)
        topology = _set_nested(topology, f"{topo2}.alive_mask", fc2_a)

        # Bias for fc1 also needs permuting (d_ff dimension)
        bias_path = fc1_path.replace(".values", ".bias")
        try:
            fc1_bias = _get_nested(params, bias_path)
            B = tile_size
            # Expand perm from tile-level to element-level
            elem_perm = jnp.repeat(perm * B, B) + jnp.tile(jnp.arange(B), len(perm))
            params = _set_nested(params, bias_path, fc1_bias[elem_perm])
        except (KeyError, IndexError):
            pass

        ka = compute_k_active_macros(fc1_a, fc1_c, tiles_per_macro, C_ff_tiles)
        k_active_macros[layer] = int(ka)
        print(f"  {layer}: K_active_macros={int(ka)}")

    return params, topology, k_active_macros


def rebuild_optimizer(
    tx,
    new_params: Any,
) -> Any:
    """Create fresh optimizer state for the compacted params.

    We lose momentum/variance from pre-compaction, but this is intentional:
    the parameter shapes changed, and the old momentum for dead tiles is
    meaningless. The optimizer warms up quickly on the compact model.
    """
    return tx.init(new_params)


def add_routing_params(
    params: Any,
    cfg,
    key: jax.Array,
) -> Tuple[Any, jnp.ndarray]:
    """Add iteration embedding + router parameters for Phase C.

    Returns (updated params, iteration_embedding_weight).
    """
    # Iteration embedding: [max_depth, d_model], near-zero init
    iter_embed = jax.random.normal(key, (cfg.max_depth, cfg.d_model)) * 0.001
    # This gets added to the model's params under a known path
    # For now, return it separately — the LoopedTransformer picks it up
    # via its n_clusters > 0 condition
    return params, iter_embed


def _log_live_arrays(label: str) -> None:
    """Log count and total bytes of live JAX arrays (best-effort)."""
    try:
        arrays = jax.live_arrays()
        total_bytes = sum(a.nbytes for a in arrays)
        print(f"  [{label}] live JAX arrays: {len(arrays)}, "
              f"total: {total_bytes / 1024**3:.2f} GiB")
    except Exception as e:
        print(f"  [{label}] live_arrays() unavailable: {e}")


def full_compaction(
    params: Any,
    topology: Any,
    bell_paths: List[Tuple[str, str, str]],
    tx,
    cfg,
    key: jax.Array,
) -> Tuple[Any, Any, Any, Dict[str, int]]:
    """Complete compaction pipeline.

    1. Pack alive tiles, shrink K
    2. Column reorder for macro-block density
    3. Rebuild optimizer
    4. Add routing params
    5. Free old state, flush XLA cache, GC

    Returns (params, topology, opt_state, k_active_macros)
    """
    print(f"\n{'─'*60}")
    print("Phase B → C: Compaction")
    print(f"{'─'*60}")

    _log_live_arrays("pre-compact")

    # 1. Compact: shrink K
    print("\n1. Packing alive tiles...")
    old_params = params
    old_topology = topology
    params, topology, k_new_map = compact_params_and_topology(
        params, topology, bell_paths
    )
    # Drop references to old pytrees so GC can collect them
    del old_params, old_topology

    # 2. Column reorder
    print("\n2. Column reorder for macro-block density...")
    params, topology, k_active_macros = column_reorder_all(
        params, topology, bell_paths,
        n_core=cfg.n_core,
        macro_tile_size=cfg.macro_tile_size,
        tile_size=cfg.tile_size,
    )

    # 3. Rebuild optimizer (fresh state for new shapes).
    #    Explicitly delete old state before creating new one so XLA can
    #    release the backing buffers before the new allocation.
    print("\n3. Rebuilding optimizer state...")
    opt_state = rebuild_optimizer(tx, params)

    # 4. Routing params
    print("\n4. Adding routing params...")
    params, iter_embed = add_routing_params(params, cfg, key)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nCompaction complete: {total_params:,} params")
    print(f"K_active_macros: {k_active_macros}")
    print(f"{'─'*60}\n")

    # 5. Free XLA compiled-program cache (cached programs reference old shapes)
    #    and force Python GC to reclaim unreachable buffers.
    print("5. Flushing XLA cache and running GC...")
    jax.clear_caches()
    gc.collect()
    _log_live_arrays("post-compact")

    return params, topology, opt_state, k_active_macros


# ---------------------------------------------------------------------------
# Pytree helpers (used by train.py too)
# ---------------------------------------------------------------------------

def _get_nested(d, path):
    for key in path.split("."):
        d = d[key]
    return d


def _set_nested(d, path, value):
    keys = path.split(".")
    if len(keys) == 1:
        return {**d, keys[0]: value}
    return {**d, keys[0]: _set_nested(d[keys[0]], ".".join(keys[1:]), value)}
