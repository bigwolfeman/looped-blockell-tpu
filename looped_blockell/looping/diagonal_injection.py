"""SSM-style diagonal injection for stable looped transformers (JAX/Flax).

Ports titans_core/looping/diagonal_injection.py to JAX/Flax.

Implements the state transition from Parcae (arXiv:2604.12946).
Guarantees spectral radius < 1 by construction via negative diagonal
parameterization with Zero-Order Hold (ZOH) discretization:

    A   = -exp(log_A)          # forced negative
    dt  = softplus(log_dt)     # forced positive
    decay = exp(dt * A)        # ∈ (0, 1) guaranteed
    h_new = decay * h + dt * e
"""

import math
import jax
import jax.numpy as jnp
import flax.linen as nn


class DiagonalInjection(nn.Module):
    """SSM-style diagonal state transition for looped transformers.

    Blends recurrent hidden state with input embeddings using learned
    per-dimension decay rates. Stability is guaranteed by construction:
    all decay values lie strictly in (0, 1).

    Attributes:
        d_model:    Hidden dimension. State dim = d_model (no B/C projections).
        init_decay: Target initial decay per step.
                    Default 0.447 ≈ sqrt(1/5) from Parcae.
                    At init: log_A=0 → A=-1, decay = exp(-dt).
                    Want decay=init_decay → dt = -log(init_decay).
    """

    d_model: int
    init_decay: float = 0.447

    @nn.compact
    def __call__(self, h: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
        """Blend state h with input e via learned per-dim decay.

        Args:
            h: Recurrent hidden state  [B, S, d_model]
            e: Input embeddings        [B, S, d_model]

        Returns:
            Updated state              [B, S, d_model]
        """
        # log_A: initialized to 0 → A = -exp(0) = -1 at init
        log_A = self.param(
            "log_A",
            nn.initializers.zeros,
            (self.d_model,),
        )

        # log_dt: initialized so that softplus(log_dt) = -log(init_decay)
        # softplus(x) = log(1 + exp(x))  →  x = log(exp(target) - 1)
        target_dt = -math.log(self.init_decay)          # = -log(0.447) ≈ 0.806
        init_log_dt_val = math.log(math.exp(target_dt) - 1.0)  # ≈ 0.533

        log_dt = self.param(
            "log_dt",
            nn.initializers.constant(init_log_dt_val),
            (self.d_model,),
        )

        A = -jnp.exp(log_A)                   # (−∞, 0)
        dt = jax.nn.softplus(log_dt)           # (0, +∞)
        decay = jnp.exp(dt * A)                # (0, 1) — spectral radius < 1

        return decay * h + dt * e

    def get_stats(self, params: dict) -> dict:
        """Return injection statistics for logging (call outside jit).

        Args:
            params: Flattened param dict containing 'log_A' and 'log_dt'.

        Returns:
            dict with decay_mean, decay_min, decay_max, dt_mean,
            spectral_radius.
        """
        log_A = params["log_A"]
        log_dt = params["log_dt"]
        A = -jnp.exp(log_A)
        dt = jax.nn.softplus(log_dt)
        decay = jnp.exp(dt * A)
        return {
            "decay_mean": float(decay.mean()),
            "decay_min": float(decay.min()),
            "decay_max": float(decay.max()),
            "dt_mean": float(dt.mean()),
            "spectral_radius": float(jnp.abs(decay).max()),
        }
