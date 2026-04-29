"""Lorentz hyperboloid and hybrid embeddings for JAX/Flax.

Ported from BLT-Jepa token_jepa/lorentz.py.

Lorentz embeddings live on H^d = {x in R^(d+1) : <x,x>_L = -1, x_0 > 0}.
Transformer ops happen in tangent space at origin o = (1, 0, ..., 0).
"""

import math
import jax
import jax.numpy as jnp
import flax.linen as nn

EPS = 1e-6


def minkowski_dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(axis=-1, keepdims=True)


def project_to_hyperboloid(space: jnp.ndarray) -> jnp.ndarray:
    sq_norm = (space * space).sum(axis=-1, keepdims=True)
    x0 = jnp.sqrt(jnp.maximum(1.0 + sq_norm, EPS))
    return jnp.concatenate([x0, space], axis=-1)


def _safe_acosh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.arccosh(jnp.maximum(x, 1.0 + EPS))


def log_map_origin(x: jnp.ndarray) -> jnp.ndarray:
    """H^d → tangent space at origin. Returns d-dim (drops time component)."""
    x0 = x[..., :1]
    xs = x[..., 1:]
    alpha = _safe_acosh(x0)
    denom = jnp.sqrt(jnp.maximum(x0 * x0 - 1.0, EPS))
    coeff = jnp.where(denom < 1e-4, jnp.ones_like(denom), alpha / denom)
    return coeff * xs


def exp_map_origin(v: jnp.ndarray) -> jnp.ndarray:
    """Tangent space at origin → H^d. v is d-dim space components."""
    norm = jnp.sqrt(jnp.maximum((v * v).sum(axis=-1, keepdims=True), EPS))
    x0 = jnp.cosh(norm)
    coeff = jnp.where(norm < 1e-4, jnp.ones_like(norm), jnp.sinh(norm) / norm)
    return jnp.concatenate([x0, coeff * v], axis=-1)


def lorentz_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return _safe_acosh(-minkowski_dot(x, y))


class LorentzEmbedding(nn.Module):
    """Embedding on the Lorentz hyperboloid.

    Stores d-dim space components, computes time component on the fly.
    Returns d-dim tangent vectors at origin for transformer processing.
    """
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        space_embed = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            embedding_init=nn.initializers.normal(stddev=0.005),
            name="space_embed",
        )
        space = space_embed(input_ids)
        hyp = project_to_hyperboloid(space)
        return log_map_origin(hyp)

    def attend(self, x: jnp.ndarray) -> jnp.ndarray:
        """Weight-tied LM head: project space weights to tangent, then dot."""
        space_w = self.variables["params"]["space_embed"]["embedding"]
        hyp = project_to_hyperboloid(space_w)
        tangent_w = log_map_origin(hyp)  # [vocab, features]
        return x @ tangent_w.T


class HybridEmbedding(nn.Module):
    """Split embedding: part Euclidean, part Lorentz, concatenated.

    Total output dim = euclidean_dim + lorentz_dim.
    """
    num_embeddings: int
    euclidean_dim: int
    lorentz_dim: int

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        euc = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.euclidean_dim,
            embedding_init=nn.initializers.normal(stddev=1.0 / math.sqrt(self.euclidean_dim)),
            name="euc_embed",
        )(input_ids)

        lor = LorentzEmbedding(
            num_embeddings=self.num_embeddings,
            features=self.lorentz_dim,
            name="lor_embed",
        )(input_ids)

        return jnp.concatenate([euc, lor], axis=-1)

    def attend(self, x: jnp.ndarray) -> jnp.ndarray:
        """Weight-tied LM head for hybrid embeddings."""
        euc_w = self.variables["params"]["euc_embed"]["embedding"]  # [vocab, euc_dim]

        lor_space_w = self.variables["params"]["lor_embed"]["space_embed"]["embedding"]
        lor_hyp = project_to_hyperboloid(lor_space_w)
        lor_tangent_w = log_map_origin(lor_hyp)  # [vocab, lor_dim]

        full_w = jnp.concatenate([euc_w, lor_tangent_w], axis=-1)  # [vocab, total_dim]
        return x @ full_w.T
