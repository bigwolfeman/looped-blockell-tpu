"""CMS scoring, topology management, and column reordering."""

from .cms import (
    CMSState,
    init_cms_state,
    accumulate_scores,
    score_step,
    prune_step,
    get_density,
    get_alive_count,
)

from .column_reorder import (
    compute_column_importance,
    compute_permutation,
    apply_permutation_to_block_ell,
    apply_permutation_to_optimizer,
    compute_k_active_macros,
    full_reorder_step,
)

from .tile_pruning import (
    create_tile_masks,
    apply_tile_masks,
    init_tile_scores,
    accumulate_tile_scores,
    normalize_tile_scores,
    prune_tiles,
    TileScores,
)

__all__ = [
    # cms.py (Block-ELL format scoring)
    "CMSState",
    "init_cms_state",
    "accumulate_scores",
    "score_step",
    "prune_step",
    "get_density",
    "get_alive_count",
    # tile_pruning.py (dense weight tile masking)
    "create_tile_masks",
    "apply_tile_masks",
    "init_tile_scores",
    "accumulate_tile_scores",
    "normalize_tile_scores",
    "prune_tiles",
    "TileScores",
    # column_reorder.py
    "compute_column_importance",
    "compute_permutation",
    "apply_permutation_to_block_ell",
    "apply_permutation_to_optimizer",
    "compute_k_active_macros",
    "full_reorder_step",
]
