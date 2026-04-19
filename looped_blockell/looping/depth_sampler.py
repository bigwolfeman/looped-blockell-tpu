"""Per-sequence depth sampling for looped transformers (JAX/Flax).

Ports titans_core/looping/depth_sampler.py to JAX.

Implements Poisson depth sampling from Parcae (arXiv:2604.12946),
using the "poisson-truncated-full" scheme:

    total_i ~ Poisson(mean_depth), clamped to [min_depth, max_depth]
    k_i = min(total_i, bptt_depth)    # grad iterations (at end)
    n_i = total_i - k_i               # no-grad iterations (at start)

The batch runs max(n_i) no-grad steps, then max(k_i) grad steps.
Finished sequences are FROZEN via jnp.where — NOT zeroed.

NOTE on JAX static shapes:
    n_max / k_max are Python ints. They must be resolved BEFORE entering
    jit so lax.scan gets a static iteration count. Pass them to the
    jitted function as static_argnums or extract them before the jit call.
    See LoopedTransformer for the canonical usage pattern.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class DepthPlan:
    """Per-sequence iteration plan for a batch.

    All tensor fields are JAX arrays (not NumPy). n_max / k_max are plain
    Python ints so they can be used as static values in lax.scan.
    """

    total: jnp.ndarray     # [B] int32 — total iterations per sequence
    n_nograd: jnp.ndarray  # [B] int32 — no-grad (stop-grad) iters per sequence
    k_grad: jnp.ndarray    # [B] int32 — grad iters per sequence
    n_max: int             # Python int — max no-grad steps to run for the batch
    k_max: int             # Python int — max grad steps to run for the batch

    @property
    def t_max(self) -> int:
        """Total iterations to run (n_max + k_max)."""
        return self.n_max + self.k_max


def _resolve_bptt(mean_depth: int, bptt_depth: int | None) -> int:
    """Default bptt to ceil(mean_depth / 2) when not specified."""
    return bptt_depth if bptt_depth is not None else -(-mean_depth // 2)


def sample_depth(
    key: jax.Array,
    batch_size: int,
    mean_depth: int,
    min_depth: int = 1,
    max_depth: int = 32,
    bptt_depth: int | None = None,
) -> DepthPlan:
    """Sample per-sequence depths using poisson-truncated-full scheme.

    Each sequence i gets:
        total_i = clamp(Poisson(mean_depth), min_depth, max_depth)
        k_i     = min(total_i, bptt_depth)   — grad steps (at end)
        n_i     = total_i - k_i               — no-grad steps (at start)

    The returned n_max / k_max are Python ints (max over the batch).
    Because they are computed via int(jnp.array.max()), this triggers a
    device→host transfer — call this *before* jit if inside a jitted region.

    Args:
        key:        JAX PRNG key for Poisson sampling.
        batch_size: Number of sequences in the batch.
        mean_depth: Poisson lambda (mean loop iterations).
        min_depth:  Lower clamp for sampled depths.
        max_depth:  Upper clamp for sampled depths.
        bptt_depth: Max grad iterations per sequence. Default: ceil(mean/2).

    Returns:
        DepthPlan with per-sequence arrays and Python int batch maxima.
    """
    bptt = _resolve_bptt(mean_depth, bptt_depth)

    # jax.random.poisson returns int32
    total = jax.random.poisson(key, lam=mean_depth, shape=(batch_size,))
    total = jnp.clip(total, min_depth, max_depth).astype(jnp.int32)

    k_grad = jnp.minimum(total, bptt)
    n_nograd = total - k_grad

    # Materialize to Python ints — host sync, must be outside jit
    n_max = int(n_nograd.max())
    k_max = int(k_grad.max())

    return DepthPlan(
        total=total,
        n_nograd=n_nograd,
        k_grad=k_grad,
        n_max=n_max,
        k_max=k_max,
    )


def sample_fixed(
    batch_size: int,
    mean_depth: int,
    bptt_depth: int | None = None,
) -> DepthPlan:
    """Fixed depth for all sequences — used at eval / debugging.

    No PRNG key required. Returns a deterministic DepthPlan where every
    sequence runs exactly mean_depth iterations (split into nograd + grad).

    Args:
        batch_size: Number of sequences.
        mean_depth: Fixed loop depth for all sequences.
        bptt_depth: Max grad iterations per sequence. Default: ceil(mean/2).

    Returns:
        DepthPlan with all sequences at mean_depth.
    """
    bptt = _resolve_bptt(mean_depth, bptt_depth)

    total = jnp.full((batch_size,), mean_depth, dtype=jnp.int32)
    k_grad = jnp.minimum(total, bptt)
    n_nograd = total - k_grad

    return DepthPlan(
        total=total,
        n_nograd=n_nograd,
        k_grad=k_grad,
        n_max=int(n_nograd.max()),
        k_max=int(k_grad.max()),
    )
