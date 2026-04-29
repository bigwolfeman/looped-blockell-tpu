"""Loop-boundary hyper-connections (Hyperloop-style).

Replaces single-stream residual with n-stream mixing at loop boundaries.
Uses diagonal parameterization (not Sinkhorn) per Hyperloop finding that
diagonal beats doubly-stochastic for looped transformers.

Architecture:
    Before each loop iteration:
        h_expanded = diag(alpha) @ h_streams    (n streams, each d_model)
        h_input    = sum(h_expanded, axis=0)     (aggregate to d_model for layer)

    After each loop iteration:
        h_output   = layer(h_input)              (d_model)
        h_streams  = diag(beta_res) @ h_streams + diag(beta_out) @ broadcast(h_output)

Where alpha, beta_res, beta_out are learned [n_streams] vectors per loop boundary.
Initialized near identity: alpha=1/n, beta_res≈1, beta_out≈1/n.

Reference: arXiv:2604.21254 (Hyperloop Transformers, MIT, Apr 2026)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class LoopBoundaryHC(nn.Module):
    """Hyper-connection module for loop boundaries.

    Manages n parallel residual streams. Called at the start and end of
    each loop iteration to mix streams and distribute layer output.
    """
    d_model: int
    n_streams: int = 4

    @nn.compact
    def __call__(
        self,
        streams: jnp.ndarray,
        layer_output: jnp.ndarray | None = None,
        mode: str = "aggregate",
    ) -> jnp.ndarray:
        """
        Args:
            streams: [B, S, n_streams, d_model] — parallel residual streams
            layer_output: [B, S, d_model] — output from core layers (only for "distribute")
            mode: "aggregate" (streams → single input) or "distribute" (output → streams)

        Returns:
            "aggregate": [B, S, d_model] — aggregated input for core layers
            "distribute": [B, S, n_streams, d_model] — updated streams
        """
        n = self.n_streams

        if mode == "aggregate":
            alpha = self.param(
                "alpha",
                nn.initializers.constant(1.0 / n),
                (n,),
            )
            # alpha[i] weights stream i, then sum → single d_model vector
            weighted = streams * alpha[None, None, :, None]  # [B, S, n, d]
            return weighted.sum(axis=2)  # [B, S, d]

        elif mode == "distribute":
            beta_res = self.param(
                "beta_res",
                nn.initializers.ones,
                (n,),
            )
            beta_out = self.param(
                "beta_out",
                nn.initializers.constant(1.0 / n),
                (n,),
            )
            # Residual: scale each existing stream
            residual = streams * beta_res[None, None, :, None]
            # Output: broadcast layer output to all streams with per-stream scaling
            broadcast = layer_output[:, :, None, :] * beta_out[None, None, :, None]
            return residual + broadcast

        else:
            raise ValueError(f"Unknown mode: {mode}")


def init_streams(h: jnp.ndarray, n_streams: int) -> jnp.ndarray:
    """Initialize n parallel streams from a single hidden state.

    Copies h into all n streams (identity-like initialization).
    """
    return jnp.broadcast_to(
        h[:, :, None, :],
        (*h.shape[:2], n_streams, h.shape[-1]),
    ).copy()


def collapse_streams(streams: jnp.ndarray) -> jnp.ndarray:
    """Collapse n streams back to single hidden state via mean."""
    return streams.mean(axis=2)
