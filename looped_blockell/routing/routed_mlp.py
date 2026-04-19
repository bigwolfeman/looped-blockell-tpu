"""RoutedMLP: MLP with per-token tile-group routing via ReMoE.

Architecture
------------
Dense path (use_block_sparse=False, Phase B.1):
    x → fc1 (d_model → d_ff, Dense) → GELU → fc2 (d_ff → d_model, Dense)

Routed path (use_block_sparse=True, Phase C):
    x  → fc1_router → gates_fc1 [B, T, n_clusters]
    x  → fc1 Dense  → hidden [B, T, d_ff]
    hidden = hidden * gates_expanded          (cluster gating)
    hidden = GELU(hidden)
    hidden → fc2_router → gates_fc2
    output = fc2(hidden * fc2_gates_expanded) → [B, T, d_model]

The per-cluster gate is broadcast over all d_ff features within each cluster
(cluster_size = d_ff // n_clusters features per cluster).  This mirrors the
block-sparse view where a cluster corresponds to a tile-group.

L1 loss
-------
The module returns (output, l1_loss) where l1_loss should be added to the
main task loss *scaled by a small coefficient* (e.g., 1e-3).  The training
loop must call routing.remoe_router.update_lambda() after each step to adapt
the per-cluster L1 coefficients.

Note on lambda_ state
---------------------
lambda_ is NOT a Flax variable — it is external state carried in the training
loop alongside the optimizer state, matching PyTorch convention where it was a
plain tensor.  Pass it into __call__; update it outside the module.

Usage::

    model = RoutedMLP(d_model=512, d_ff=2048, n_clusters=16)
    params = model.init(key, x, lambda_, deterministic=True)['params']

    (output, l1_loss), updates = model.apply(
        {'params': params},
        x,
        lambda_,
        deterministic=False,
        rngs={'dropout': dropout_key},
        mutable=[],
    )
    total_loss = task_loss + 1e-3 * l1_loss
    lambda_ = update_lambda(lambda_, gates, target_sparsity=model.target_sparsity)
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .remoe_router import ReMoERouter, compute_l1_loss


class RoutedMLP(nn.Module):
    """Two-layer MLP with optional ReMoE per-token tile-group routing.

    Attributes
    ----------
    d_model : int
    d_ff : int
    n_clusters : int        — number of routing clusters (tile-groups)
    d_query : int           — router hidden dimension
    target_sparsity : float — target fraction of dead clusters per token
    alpha : float           — λ update multiplicative step
    dropout : float         — dropout rate on output (0 = no dropout)
    use_block_sparse : bool — if False, dense path (Phase B.1); if True, routed path
    dtype : dtype           — compute dtype (default bfloat16)
    """

    d_model: int
    d_ff: int
    n_clusters: int = 16
    d_query: int = 256
    target_sparsity: float = 0.5
    alpha: float = 1.2
    dropout: float = 0.0
    use_block_sparse: bool = False
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                   # [B, T, d_model]
        lambda_: jnp.ndarray,              # [n_clusters] external L1 coefficients
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Parameters
        ----------
        x : [B, T, d_model]
        lambda_ : [n_clusters] float32   — L1 penalty coefficients (external state)
        deterministic : bool             — disables dropout when True

        Returns
        -------
        output : [B, T, d_model]
        l1_loss : scalar — sum of ReMoE L1 losses for fc1 + fc2 routers
        """
        if self.use_block_sparse:
            return self._forward_routed(x, lambda_, deterministic)
        return self._forward_dense(x, deterministic)

    # ------------------------------------------------------------------
    # Dense path (Phase B.1)
    # ------------------------------------------------------------------

    def _forward_dense(
        self,
        x: jnp.ndarray,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        hidden = nn.Dense(self.d_ff, dtype=self.dtype, name="fc1")(x)
        hidden = jax.nn.gelu(hidden)
        output = nn.Dense(self.d_model, dtype=self.dtype, name="fc2")(hidden)
        if self.dropout > 0.0:
            output = nn.Dropout(rate=self.dropout)(output, deterministic=deterministic)
        l1_loss = jnp.array(0.0, dtype=jnp.float32)
        return output, l1_loss

    # ------------------------------------------------------------------
    # Routed path (Phase C)
    # ------------------------------------------------------------------

    def _forward_routed(
        self,
        x: jnp.ndarray,                   # [B, T, d_model]
        lambda_: jnp.ndarray,              # [n_clusters]
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cluster_size_fc1 = self.d_ff // self.n_clusters
        cluster_size_fc2 = self.d_ff // self.n_clusters  # fc2 gates on hidden (d_ff) clusters

        # --- fc1 routing ---------------------------------------------------
        gates_fc1 = ReMoERouter(
            n_clusters=self.n_clusters,
            d_query=self.d_query,
            target_sparsity=self.target_sparsity,
            alpha=self.alpha,
            dtype=self.dtype,
            name="fc1_router",
        )(x)  # [B, T, n_clusters]

        # Dense fc1 matmul
        hidden = nn.Dense(self.d_ff, use_bias=True, dtype=self.dtype, name="fc1")(x)
        # [B, T, d_ff]

        # Expand cluster gates to d_ff features
        # gates_fc1: [B, T, n_clusters] → [B, T, n_clusters * cluster_size_fc1]
        gates_fc1_expanded = jnp.repeat(gates_fc1, cluster_size_fc1, axis=-1)  # [B, T, d_ff]

        # Cast gates to hidden dtype before multiplication
        hidden = hidden * gates_fc1_expanded.astype(hidden.dtype)

        # L1 loss for fc1 router (in float32 for stability)
        l1_fc1 = compute_l1_loss(gates_fc1.astype(jnp.float32), lambda_)

        # --- Non-linearity -------------------------------------------------
        hidden = jax.nn.gelu(hidden)

        # --- fc2 routing ---------------------------------------------------
        # Route on the gated hidden (after GELU) — captures what actually flows
        gates_fc2 = ReMoERouter(
            n_clusters=self.n_clusters,
            d_query=self.d_query,
            target_sparsity=self.target_sparsity,
            alpha=self.alpha,
            dtype=self.dtype,
            name="fc2_router",
        )(hidden)  # [B, T, n_clusters]

        # Gate the hidden before fc2: controls which d_ff clusters contribute
        gates_fc2_expanded = jnp.repeat(gates_fc2, cluster_size_fc2, axis=-1)  # [B, T, d_ff]
        hidden_gated = hidden * gates_fc2_expanded.astype(hidden.dtype)

        # Dense fc2 matmul
        output = nn.Dense(self.d_model, use_bias=True, dtype=self.dtype, name="fc2")(hidden_gated)
        # [B, T, d_model]

        # L1 loss for fc2 router
        l1_fc2 = compute_l1_loss(gates_fc2.astype(jnp.float32), lambda_)

        if self.dropout > 0.0:
            output = nn.Dropout(rate=self.dropout)(output, deterministic=deterministic)

        l1_total = l1_fc1 + l1_fc2
        return output, l1_total


# ---------------------------------------------------------------------------
# Convenience: extract gates for external lambda_ update
# ---------------------------------------------------------------------------

class RoutedMLPWithGates(nn.Module):
    """RoutedMLP variant that also returns raw gates for lambda_ updates.

    The training loop needs the raw gates to call update_lambda().
    This wrapper calls RoutedMLP and additionally re-runs the routers to
    capture gates — only useful during training.

    In practice, prefer using RoutedMLP and capturing gates inside a custom
    training step that threads them out via a scan carry or auxiliary output.

    Attributes  — same as RoutedMLP.
    """

    d_model: int
    d_ff: int
    n_clusters: int = 16
    d_query: int = 256
    target_sparsity: float = 0.5
    alpha: float = 1.2
    dropout: float = 0.0
    use_block_sparse: bool = False
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        lambda_: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns (output, l1_loss, gates_fc1, gates_fc2).

        gates_fc1, gates_fc2 : [B, T, n_clusters] — for update_lambda().
        """
        mlp = RoutedMLP(
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_clusters=self.n_clusters,
            d_query=self.d_query,
            target_sparsity=self.target_sparsity,
            alpha=self.alpha,
            dropout=self.dropout,
            use_block_sparse=self.use_block_sparse,
            dtype=self.dtype,
            name="mlp",
        )
        output, l1_loss = mlp(x, lambda_, deterministic)

        if self.use_block_sparse:
            # Re-run just the routers to capture gates (shares params via name)
            gates_fc1 = ReMoERouter(
                n_clusters=self.n_clusters,
                d_query=self.d_query,
                target_sparsity=self.target_sparsity,
                alpha=self.alpha,
                dtype=self.dtype,
                name="mlp/fc1_router",  # shared name with inner router
            )(x)
            # For fc2 gates we need the intermediate hidden — recompute cheaply
            hidden = nn.Dense(self.d_ff, use_bias=True, dtype=self.dtype, name="mlp/fc1")(x)
            cluster_size = self.d_ff // self.n_clusters
            gates_fc1_exp = jnp.repeat(gates_fc1, cluster_size, axis=-1).astype(hidden.dtype)
            hidden = jax.nn.gelu(hidden * gates_fc1_exp)
            gates_fc2 = ReMoERouter(
                n_clusters=self.n_clusters,
                d_query=self.d_query,
                target_sparsity=self.target_sparsity,
                alpha=self.alpha,
                dtype=self.dtype,
                name="mlp/fc2_router",
            )(hidden)
        else:
            gates_fc1 = jnp.zeros((x.shape[0], x.shape[1], self.n_clusters), dtype=jnp.float32)
            gates_fc2 = jnp.zeros_like(gates_fc1)

        return output, l1_loss, gates_fc1, gates_fc2
