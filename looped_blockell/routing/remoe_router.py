"""ReMoE router: ReLU gates + adaptive per-cluster L1 regularisation.

Reference: "ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing"

Key properties
--------------
* ReLU gates yield hard zeros (true sparsity) while remaining differentiable.
* Per-cluster L1 coefficients λ_c are updated via a zeroth-order multiplicative
  rule that drives each cluster toward the target_sparsity independently.
* Load balancing emerges naturally without auxiliary losses.

Usage::

    router = ReMoERouter(n_clusters=16, d_query=256, target_sparsity=0.5)
    params = router.init(key, x)['params']

    gates, lambda_ = router.apply(
        {'params': params},
        x,                     # [B, T, d_input]
        lambda_,               # [n_clusters] L1 coefficients (carried state)
    )
    # gates: [B, T, n_clusters]   — non-negative, sparse via ReLU

    # L1 loss to add to total loss:
    loss += compute_l1_loss(gates, lambda_)

    # After optimiser step, update lambda:
    lambda_ = update_lambda(lambda_, gates, target_sparsity=0.5)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Router module
# ---------------------------------------------------------------------------

class ReMoERouter(nn.Module):
    """Two-layer MLP router producing ReLU-sparse cluster gates.

    Architecture: x → Linear(d_query) → ReLU → Linear(n_clusters) → ReLU

    The inner ReLU gives the hidden projection sparse activations for
    efficiency; the outer ReLU produces the final non-negative gates which
    are exactly zero for inactive clusters.

    Attributes
    ----------
    n_clusters : int    — number of tile-groups / expert clusters
    d_query : int       — hidden dimension of routing MLP
    target_sparsity : float  — fraction of clusters that should be inactive
                               (used externally by update_lambda)
    alpha : float       — multiplicative step size for λ updates

    Forward
    -------
    x : [B, T, d_input]
    → gates : [B, T, n_clusters]  non-negative, ReLU-sparse
    """

    n_clusters: int = 16
    d_query: int = 256
    target_sparsity: float = 0.5
    alpha: float = 1.2
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, T, d_input] → gates: [B, T, n_clusters]."""
        # Project to query space and apply inner ReLU for routing sparsity
        h = nn.Dense(
            self.d_query,
            use_bias=True,
            dtype=self.dtype,
            name="gate_proj",
        )(x)
        h = nn.relu(h)

        # Produce raw gate logits and clip to non-negative via ReLU
        gates = nn.Dense(
            self.n_clusters,
            use_bias=True,
            dtype=self.dtype,
            name="gate_out",
        )(h)
        gates = nn.relu(gates)

        return gates


# ---------------------------------------------------------------------------
# L1 load-balancing loss
# ---------------------------------------------------------------------------

def compute_l1_loss(
    gates: jnp.ndarray,    # [B, T, n_clusters]
    lambda_: jnp.ndarray,  # [n_clusters]
) -> jnp.ndarray:
    """Per-cluster L1 regularisation for load balancing.

    Penalises the mean gate activation of each cluster, weighted by its
    current λ coefficient.  When λ_c is high the cluster is forced sparser;
    when λ_c is low the cluster is allowed to activate freely.

    Formula
    -------
    f_c = fraction of (B, T) positions where gate_c > 0
    L1 = Σ_c  λ_c * f_c * mean_gate_c

    Parameters
    ----------
    gates : [B, T, n_clusters]
    lambda_ : [n_clusters]  — per-cluster L1 coefficients (>= 0)

    Returns
    -------
    Scalar loss value.
    """
    gates = gates.astype(jnp.float32)

    # Fraction of positions where each cluster fires: [n_clusters]
    f = (gates > 0).mean(axis=(0, 1))

    # Mean gate value per cluster: [n_clusters]
    mean_gate = gates.mean(axis=(0, 1))

    # Weighted L1
    weighted = lambda_.astype(jnp.float32) * f * mean_gate
    return weighted.sum()


# ---------------------------------------------------------------------------
# Lambda update rule
# ---------------------------------------------------------------------------

def update_lambda(
    lambda_: jnp.ndarray,      # [n_clusters]
    gates: jnp.ndarray,         # [B, T, n_clusters]
    target_sparsity: float = 0.5,
    alpha: float = 1.2,
    lambda_min: float = 0.01,
    lambda_max: float = 100.0,
) -> jnp.ndarray:
    """Zeroth-order multiplicative update of per-cluster L1 coefficients.

    For each cluster c:
      - sparsity_c = fraction of positions where gate_c == 0
      - sign_c = sign(sparsity_c - target_sparsity)
        * positive → cluster is too active → increase λ (push more zeros)
        * negative → cluster is too sparse → decrease λ (allow more activation)
      - new_λ_c = clip(λ_c * alpha^sign_c, lambda_min, lambda_max)

    Parameters
    ----------
    lambda_ : [n_clusters]
    gates   : [B, T, n_clusters]
    target_sparsity : float
    alpha : float — multiplicative step (>1, e.g. 1.2)
    lambda_min, lambda_max : float — clamp bounds

    Returns
    -------
    new_lambda_ : [n_clusters]
    """
    gates = gates.astype(jnp.float32)

    # Fraction of positions with zero gate per cluster [n_clusters]
    sparsity = (gates == 0).astype(jnp.float32).mean(axis=(0, 1))

    # Sign of deviation from target
    sign = jnp.sign(sparsity - target_sparsity)

    # Multiplicative update
    new_lambda = lambda_.astype(jnp.float32) * (alpha ** sign)
    return jnp.clip(new_lambda, lambda_min, lambda_max)


# ---------------------------------------------------------------------------
# Convenience initialiser
# ---------------------------------------------------------------------------

def init_lambda(n_clusters: int, init_value: float = 1.0) -> jnp.ndarray:
    """Initialise per-cluster L1 coefficients to a uniform value.

    Parameters
    ----------
    n_clusters : int
    init_value : float — starting λ (1.0 is a reasonable default)

    Returns
    -------
    lambda_ : [n_clusters] float32
    """
    return jnp.full((n_clusters,), init_value, dtype=jnp.float32)
