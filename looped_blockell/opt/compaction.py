"""Compaction: masked-dense → Block-ELL with column reorder + d_ff truncation.

Called once at the prune→route transition (end of Phase B, start of Phase C).

Steps:
1. For each core MLP, extract alive tiles from dense weights → Block-ELL format
2. Column-reorder d_ff to cluster alive tiles spatially (macro-block density)
3. Physically truncate d_ff to K_active * macro_tile_size
4. Rebuild optimizer state for the new compact shapes
5. Return compact model params + new optimizer state

The compact model runs with Block-ELL matmul (gather→MXU→scatter), which
gives actual compute savings proportional to density.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from ..layers.block_ell import BlockELLConfig, BlockELLTensor


def extract_block_ell_from_masked_dense(
    kernel: jnp.ndarray,
    alive_mask: jnp.ndarray,
    tile_size: int = 16,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, BlockELLConfig]:
    """Convert a masked dense weight to Block-ELL format.

    Only extracts alive tiles — dead tiles are discarded entirely.

    Args:
        kernel: [out_features, in_features] dense weight (dead tiles already zeroed)
        alive_mask: [R, C] bool tile mask
        tile_size: B

    Returns:
        (values, col_indices, alive_mask_compact, config)
        values: [R, K, B, B] — K = max alive tiles per row
        col_indices: [R, K] int32 — column index of each alive tile
        alive_mask_compact: [R, K] bool — all True (no dead tiles in compact format)
        config: BlockELLConfig
    """
    B = tile_size
    R, C = alive_mask.shape
    out_f, in_f = kernel.shape

    # K = max alive tiles per row (pad shorter rows)
    alive_per_row = alive_mask.sum(axis=1)  # [R]
    K = int(alive_per_row.max())

    if K == 0:
        K = 1  # degenerate: at least one tile per row

    # Reshape kernel into tiles: [R, B, C, B] → [R, C, B, B]
    tiles = kernel[:R*B, :C*B].reshape(R, B, C, B).transpose(0, 2, 1, 3)

    # For each row, gather alive tiles (pad with zeros if fewer than K)
    values = jnp.zeros((R, K, B, B), dtype=kernel.dtype)
    col_indices = jnp.full((R, K), -1, dtype=jnp.int32)
    alive_compact = jnp.zeros((R, K), dtype=jnp.bool_)

    def _extract_row(r):
        row_mask = alive_mask[r]  # [C]
        row_tiles = tiles[r]      # [C, B, B]

        # Get indices of alive tiles (sorted)
        alive_idx = jnp.where(row_mask, jnp.arange(C), C)  # dead → C (out of range)
        alive_idx = jnp.sort(alive_idx)  # alive first, then C sentinels

        # Take first K (may include sentinels if fewer than K alive)
        sel_idx = alive_idx[:K]
        sel_valid = sel_idx < C

        # Gather tiles (clip index for safety)
        safe_idx = jnp.clip(sel_idx, 0, C - 1)
        sel_tiles = row_tiles[safe_idx]  # [K, B, B]
        sel_tiles = jnp.where(sel_valid[:, None, None], sel_tiles, 0.0)

        sel_cols = jnp.where(sel_valid, safe_idx, -1).astype(jnp.int32)

        return sel_tiles, sel_cols, sel_valid

    # vmap over rows
    all_tiles, all_cols, all_valid = jax.vmap(_extract_row)(jnp.arange(R))

    config = BlockELLConfig(R=R, C=C, K=K, B=B)
    return all_tiles, all_cols, all_valid, config


def compact_mlp_layer(
    fc1_kernel: jnp.ndarray,
    fc1_bias: jnp.ndarray,
    fc2_kernel: jnp.ndarray,
    fc2_bias: jnp.ndarray,
    fc1_mask: jnp.ndarray,
    fc2_mask: jnp.ndarray,
    tile_size: int = 16,
) -> Dict[str, Any]:
    """Compact one MLP layer from masked-dense to Block-ELL.

    Returns dict with Block-ELL tensors for fc1 and fc2.
    """
    fc1_vals, fc1_cols, fc1_alive, fc1_cfg = extract_block_ell_from_masked_dense(
        fc1_kernel, fc1_mask, tile_size
    )
    fc2_vals, fc2_cols, fc2_alive, fc2_cfg = extract_block_ell_from_masked_dense(
        fc2_kernel, fc2_mask, tile_size
    )

    return {
        "fc1": {
            "values": fc1_vals,
            "col_indices": fc1_cols,
            "alive_mask": fc1_alive,
            "config": fc1_cfg,
            "bias": fc1_bias,
        },
        "fc2": {
            "values": fc2_vals,
            "col_indices": fc2_cols,
            "alive_mask": fc2_alive,
            "config": fc2_cfg,
            "bias": fc2_bias,
        },
    }


def compact_all_core_layers(
    params: Any,
    masks: Dict[str, Dict[str, jnp.ndarray]],
    n_core: int,
    tile_size: int = 16,
) -> Dict[str, Dict]:
    """Compact all core MLP layers from masked-dense to Block-ELL.

    Returns dict mapping core layer name → compact Block-ELL data.
    Does NOT modify the model params — returns separate Block-ELL state.
    """
    compact_layers = {}
    for i in range(n_core):
        name = f"core_{i}"
        if name not in params or name not in masks:
            continue

        mlp = params[name]["mlp"]
        layer_masks = masks[name]

        compact = compact_mlp_layer(
            fc1_kernel=mlp["fc1"]["kernel"] if "fc1" in mlp else mlp["fc1_w"],
            fc1_bias=mlp["fc1"]["bias"] if "fc1" in mlp else mlp.get("fc1_b", jnp.zeros(1)),
            fc2_kernel=mlp["fc2"]["kernel"] if "fc2" in mlp else mlp["fc2_w"],
            fc2_bias=mlp["fc2"]["bias"] if "fc2" in mlp else mlp.get("fc2_b", jnp.zeros(1)),
            fc1_mask=layer_masks["fc1"],
            fc2_mask=layer_masks["fc2"],
            tile_size=tile_size,
        )
        compact_layers[name] = compact

        density_fc1 = float(compact["fc1"]["alive_mask"].mean())
        density_fc2 = float(compact["fc2"]["alive_mask"].mean())
        K_fc1 = compact["fc1"]["config"].K
        K_fc2 = compact["fc2"]["config"].K
        C_fc1 = compact["fc1"]["config"].C
        C_fc2 = compact["fc2"]["config"].C
        print(f"  {name}: fc1 K={K_fc1}/{C_fc1} ({density_fc1:.1%}), "
              f"fc2 K={K_fc2}/{C_fc2} ({density_fc2:.1%})")

    return compact_layers


def compute_compact_stats(compact_layers: Dict) -> Dict[str, float]:
    """Compute overall compaction statistics."""
    total_alive = 0
    total_possible = 0
    total_params_before = 0
    total_params_after = 0

    for name, layer in compact_layers.items():
        for fc_name in ("fc1", "fc2"):
            fc = layer[fc_name]
            cfg = fc["config"]
            alive = int(fc["alive_mask"].sum())
            total = cfg.R * cfg.C
            total_alive += alive
            total_possible += total
            total_params_before += cfg.R * cfg.C * cfg.B * cfg.B
            total_params_after += cfg.R * cfg.K * cfg.B * cfg.B

    density = total_alive / max(total_possible, 1)
    compression = total_params_after / max(total_params_before, 1)

    return {
        "density": density,
        "compression_ratio": compression,
        "params_before": total_params_before,
        "params_after": total_params_after,
        "total_alive_tiles": total_alive,
        "total_tiles": total_possible,
    }


def column_reorder_compact(
    compact_layers: Dict,
    macro_tile_size: int = 128,
    tile_size: int = 16,
) -> Tuple[Dict, Dict[str, int]]:
    """Apply column reorder to compact Block-ELL layers for macro-block density.

    For each core layer:
    1. Score d_ff features by alive tile count (fc1 cols + fc2 rows)
    2. Sort by importance → permutation
    3. Remap fc1 col_indices, reorder fc2 rows
    4. Compute K_active_macros

    Returns:
        (reordered compact_layers, k_active_macros per layer)
    """
    from .column_reorder import (
        compute_column_importance,
        compute_permutation,
        apply_permutation_to_block_ell,
        compute_k_active_macros,
    )

    tiles_per_macro = macro_tile_size // tile_size
    k_active = {}

    for name, layer in compact_layers.items():
        fc1 = layer["fc1"]
        fc2 = layer["fc2"]
        C_ff = fc1["config"].C  # d_ff / tile_size

        importance = compute_column_importance(
            fc1_alive_mask=fc1["alive_mask"],
            fc2_alive_mask=fc2["alive_mask"],
            fc1_col_indices=fc1["col_indices"],
            fc2_col_indices=fc2["col_indices"],
            C_ff=C_ff,
        )
        perm = compute_permutation(importance)

        # fc1: remap col_indices (column dim)
        fc1_v, fc1_c, fc1_a = apply_permutation_to_block_ell(
            fc1["values"], fc1["col_indices"], fc1["alive_mask"],
            perm, is_column_dim=True,
        )
        fc1.update({"values": fc1_v, "col_indices": fc1_c, "alive_mask": fc1_a})

        # fc2: reorder rows (row dim = d_ff)
        fc2_v, fc2_c, fc2_a = apply_permutation_to_block_ell(
            fc2["values"], fc2["col_indices"], fc2["alive_mask"],
            perm, is_column_dim=False,
        )
        fc2.update({"values": fc2_v, "col_indices": fc2_c, "alive_mask": fc2_a})

        # K_active_macros
        ka = compute_k_active_macros(fc1_a, fc1_c, tiles_per_macro, C_ff)
        k_active[name] = int(ka)
        print(f"  {name}: K_active_macros={int(ka)} "
              f"(of {C_ff // tiles_per_macro} total macro-cols)")

    return compact_layers, k_active


def full_compaction_step(
    params: Any,
    masks: Dict,
    n_core: int,
    tile_size: int = 16,
    macro_tile_size: int = 128,
) -> Tuple[Dict, Dict[str, int], Dict[str, float]]:
    """Complete compaction pipeline.

    1. Extract Block-ELL from masked-dense
    2. Column reorder for macro-block density
    3. Compute stats

    Returns:
        (compact_layers, k_active_macros, stats)
    """
    print("Compacting masked-dense → Block-ELL...")
    compact_layers = compact_all_core_layers(params, masks, n_core, tile_size)

    print("Column reordering for macro-block density...")
    compact_layers, k_active = column_reorder_compact(
        compact_layers, macro_tile_size, tile_size
    )

    stats = compute_compact_stats(compact_layers)
    print(f"Compaction complete: {stats['density']:.1%} density, "
          f"{stats['params_before']:,} → {stats['params_after']:,} params "
          f"({stats['compression_ratio']:.1%})")

    return compact_layers, k_active, stats
