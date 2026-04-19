"""RMSNorm for JAX/Flax."""

import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * scale
