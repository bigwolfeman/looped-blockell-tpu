"""CMS scoring for Block-ELL tile importance on TPU.

Continuum Memory System: gradient-based importance scoring for Block-ELL tiles.
State is carried as a pure pytree (NamedTuple) — fully compatible with jax.jit
and lax.scan.  No Python-side mutable state.

Usage in training loop::

    cms = init_cms_state(R, K)
    # After loss.backward() equivalent (jax.grad gives grads):
    cms = accumulate_scores(cms, grad_values)
    optimizer_state = apply_updates(...)

    # Every 10 steps:
    cms = score_step(cms)

    # Every N steps (prune round):
    cms, col_indices = prune_step(cms, col_indices, prune_fraction=0.10)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class CMSState:
    """Mutable CMS state carried through training as a plain pytree.

    We use a plain class with __init__ / tree_flatten / tree_unflatten
    registered via jax.tree_util so it participates in jit/vmap correctly.

    Fields
    ------
    gradient_scores : [R, K] float32
        Accumulated (summed) gradient Frobenius norms per tile.
    block_ages : [R, K] int32
        Number of score_step() calls since this tile's last topology change.
    alive_mask : [R, K] bool
        True for tiles that are currently active.
    total_score_steps : scalar int32
        Number of times accumulate_scores() has been called since the last
        score_step() normalisation.  Used for averaging.
    """

    __slots__ = ("gradient_scores", "block_ages", "alive_mask", "total_score_steps")

    def __init__(
        self,
        gradient_scores: jnp.ndarray,
        block_ages: jnp.ndarray,
        alive_mask: jnp.ndarray,
        total_score_steps: jnp.ndarray,
    ):
        self.gradient_scores = gradient_scores
        self.block_ages = block_ages
        self.alive_mask = alive_mask
        self.total_score_steps = total_score_steps

    def replace(self, **kwargs) -> "CMSState":
        return CMSState(
            gradient_scores=kwargs.get("gradient_scores", self.gradient_scores),
            block_ages=kwargs.get("block_ages", self.block_ages),
            alive_mask=kwargs.get("alive_mask", self.alive_mask),
            total_score_steps=kwargs.get("total_score_steps", self.total_score_steps),
        )


def _cms_flatten(s: CMSState):
    children = (s.gradient_scores, s.block_ages, s.alive_mask, s.total_score_steps)
    aux = None
    return children, aux


def _cms_unflatten(aux, children):
    return CMSState(*children)


jax.tree_util.register_pytree_node(CMSState, _cms_flatten, _cms_unflatten)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_cms_state(R: int, K: int) -> CMSState:
    """Initialise all tiles alive, scores zero, ages zero.

    Parameters
    ----------
    R : int  — number of block-rows.
    K : int  — active blocks per row.

    Returns
    -------
    CMSState  with all tiles alive.
    """
    return CMSState(
        gradient_scores=jnp.zeros((R, K), dtype=jnp.float32),
        block_ages=jnp.zeros((R, K), dtype=jnp.int32),
        alive_mask=jnp.ones((R, K), dtype=jnp.bool_),
        total_score_steps=jnp.array(0, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Score accumulation
# ---------------------------------------------------------------------------

def accumulate_scores(cms_state: CMSState, grad_values: jnp.ndarray) -> CMSState:
    """Accumulate gradient Frobenius norms into tile scores.

    CRITICAL: Call between jax.grad() and the optax apply_updates(), exactly
    mirroring the PyTorch contract (between loss.backward() and optimizer.step()).

    Parameters
    ----------
    cms_state : CMSState
    grad_values : [R, K, B, B] float32
        Gradients of the Block-ELL weight values tensor.

    Returns
    -------
    CMSState  with updated gradient_scores and incremented total_score_steps.
    """
    # Frobenius norm per tile: sqrt(sum of squares over last two dims) → [R, K]
    tile_norms = jnp.sqrt(jnp.sum(grad_values.astype(jnp.float32) ** 2, axis=(-2, -1)))

    # Only alive tiles accumulate scores; dead tiles stay at their current (zeroed) value
    tile_norms = jnp.where(cms_state.alive_mask, tile_norms, 0.0)

    new_scores = cms_state.gradient_scores + tile_norms
    new_steps = cms_state.total_score_steps + jnp.array(1, dtype=jnp.int32)

    return cms_state.replace(gradient_scores=new_scores, total_score_steps=new_steps)


# ---------------------------------------------------------------------------
# Score normalisation step
# ---------------------------------------------------------------------------

def score_step(cms_state: CMSState) -> CMSState:
    """Normalise scores to per-step averages and increment block ages.

    Call every ~10 training steps.  Resets the accumulator counter so the
    next window starts fresh.

    Parameters
    ----------
    cms_state : CMSState

    Returns
    -------
    CMSState  with normalised gradient_scores, incremented block_ages,
    and reset total_score_steps.
    """
    n = jnp.maximum(cms_state.total_score_steps.astype(jnp.float32), 1.0)
    normalised = cms_state.gradient_scores / n

    # Increment age only for alive tiles
    new_ages = cms_state.block_ages + cms_state.alive_mask.astype(jnp.int32)

    return cms_state.replace(
        gradient_scores=normalised,
        block_ages=new_ages,
        total_score_steps=jnp.array(0, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Pruning step
# ---------------------------------------------------------------------------

def _prune_row(
    scores: jnp.ndarray,      # [K]
    alive: jnp.ndarray,       # [K] bool
    prune_fraction: float,
) -> jnp.ndarray:
    """Return updated alive mask for a single row.

    Strategy: among the alive tiles, mark the bottom `prune_fraction` as dead.
    Uses sort-based threshold so it's differentiable-compatible (no Python loops).
    """
    K = scores.shape[0]
    n_to_prune = jnp.maximum(
        jnp.array(1, dtype=jnp.int32),
        jnp.floor(alive.sum().astype(jnp.float32) * prune_fraction).astype(jnp.int32),
    )

    # For dead tiles, set score to +inf so they're never the "lowest alive" score
    masked_scores = jnp.where(alive, scores, jnp.inf)

    # Sort indices by score ascending; the first n_to_prune are candidates to kill
    sorted_idx = jnp.argsort(masked_scores)  # [K], ascending

    # Build a kill mask: kill sorted_idx[0..n_to_prune-1]
    kill_positions = jnp.arange(K) < n_to_prune          # [K] bool — first n_to_prune slots
    # Map sorted positions back to original indices
    kill_mask_sorted = kill_positions                      # [K]

    # Scatter: create mask in original index space
    # kill_orig[sorted_idx[i]] = kill_mask_sorted[i]
    kill_orig = jnp.zeros(K, dtype=jnp.bool_)
    kill_orig = kill_orig.at[sorted_idx].set(kill_mask_sorted)

    # Only kill tiles that are alive (don't re-kill already-dead tiles)
    kill_orig = kill_orig & alive

    new_alive = alive & ~kill_orig
    return new_alive


def prune_step(
    cms_state: CMSState,
    col_indices: jnp.ndarray,
    prune_fraction: float = 0.10,
) -> Tuple[CMSState, jnp.ndarray]:
    """Prune lowest-scoring tiles per row.

    For each row, identifies the bottom `prune_fraction` of ALIVE tiles by
    gradient score and marks them dead.  Dead tiles get col_index = -1
    (sentinel indicating unused slot — kernels must skip these).

    Uses jax.vmap over rows — no Python loops.

    Parameters
    ----------
    cms_state : CMSState
    col_indices : [R, K] int32
        Current column assignments.  Dead tiles will be set to -1.
    prune_fraction : float
        Fraction of alive tiles to prune per row (default 0.10 = 10%).

    Returns
    -------
    (updated CMSState, updated col_indices)
    """
    # vmap _prune_row over rows
    new_alive = jax.vmap(
        lambda s, a: _prune_row(s, a, prune_fraction)
    )(cms_state.gradient_scores, cms_state.alive_mask)   # [R, K]

    # Tiles that just became dead: zero their scores, mark col_index = -1
    newly_dead = cms_state.alive_mask & ~new_alive         # [R, K] bool
    new_scores = jnp.where(new_alive, cms_state.gradient_scores, 0.0)

    # Dead tiles get sentinel col index -1
    new_col_indices = jnp.where(new_alive, col_indices, jnp.full_like(col_indices, -1))

    # Reset block ages for newly dead tiles (they start fresh if re-grown)
    new_ages = jnp.where(new_alive, cms_state.block_ages, jnp.zeros_like(cms_state.block_ages))

    new_state = cms_state.replace(
        gradient_scores=new_scores,
        block_ages=new_ages,
        alive_mask=new_alive,
    )
    return new_state, new_col_indices


# ---------------------------------------------------------------------------
# Utility queries
# ---------------------------------------------------------------------------

def get_density(cms_state: CMSState) -> jnp.ndarray:
    """Return current density — fraction of tiles that are alive."""
    return cms_state.alive_mask.mean()


def get_alive_count(cms_state: CMSState) -> jnp.ndarray:
    """Return total number of alive tiles."""
    return cms_state.alive_mask.sum()
