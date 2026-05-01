"""Attention Residuals — depth-wise attention over loop iterations (arXiv:2603.15031).

Replaces additive residuals with learned selective aggregation across loop
iterations. Each iteration has a learned pseudo-query vector (zero-init for
uniform initial attention per paper spec). Keys are RMSNorm'd block outputs,
values are raw block outputs.

Two implementations:
  1. Pure JAX (portable, autograd-friendly) — used by default
  2. Pallas kernel (TPU-optimized, single kernel launch, online softmax in VMEM)

The Pallas kernel fuses RMSNorm + dot-product scoring + online softmax +
weighted sum into a single kernel. For N≤10 entries of dim d, the entire
computation fits in VMEM scratch refs with zero HBM traffic beyond reading
the buffer once.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..layers.norms import RMSNorm


# ─── Pure JAX implementation ──────────────────────────────────────────────────

def depth_attention_jax(
    entries: jnp.ndarray,
    query: jnp.ndarray,
    norm_scale: jnp.ndarray,
    norm_eps: float = 1e-5,
    n_active: int | None = None,
) -> jnp.ndarray:
    """Softmax attention over depth dimension.

    Args:
        entries:    [max_T, B, S, d] — block outputs (padded with zeros beyond n_active)
        query:      [d] — learned pseudo-query for this block index
        norm_scale: [d] — RMSNorm scale parameter
        norm_eps:   RMSNorm epsilon
        n_active:   Number of valid entries (rest are padding)

    Returns:
        [B, S, d] — softmax-weighted sum of entries
    """
    max_T = entries.shape[0]
    if n_active is None:
        n_active = max_T

    # RMSNorm the entries
    rms = jnp.sqrt(jnp.mean(entries ** 2, axis=-1, keepdims=True) + norm_eps)
    normed = entries / rms * norm_scale[None, None, None, :]

    # Compute logits: [max_T, B, S]
    logits = jnp.einsum("d, tbsd -> tbs", query, normed)

    # Mask padding entries with -inf
    mask = jnp.arange(max_T)[:, None, None] < n_active
    logits = jnp.where(mask, logits, -1e9)

    # Softmax over depth dim + weighted sum
    weights = jax.nn.softmax(logits, axis=0)
    return jnp.einsum("tbs, tbsd -> bsd", weights, entries)


# ─── Pallas kernel implementation (TPU) ──────────────────────────────────────

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    HAS_PALLAS = True
except ImportError:
    HAS_PALLAS = False


def _depth_attn_pallas_kernel(
    entries_ref,     # [max_T, BLOCK_BS, D]
    norm_scale_ref,  # [D]
    query_ref,       # [D]
    out_ref,         # [BLOCK_BS, D]
    *,
    n_active: int,
    D: int,
    norm_eps: float,
):
    """Pallas kernel: online softmax over depth dim.

    Each program instance handles BLOCK_BS positions. Iterates over N entries,
    maintaining running max/sum/output in scratch (VMEM on TPU).
    """
    q = query_ref[:]                    # [D]
    scale = norm_scale_ref[:]           # [D]

    # Initialize online softmax state
    m_prev = jnp.full((out_ref.shape[0],), -1e9, dtype=jnp.float32)  # [BLOCK_BS]
    l_prev = jnp.zeros((out_ref.shape[0],), dtype=jnp.float32)
    o_acc = jnp.zeros_like(out_ref[:], dtype=jnp.float32)            # [BLOCK_BS, D]

    def _body(t, carry):
        m_p, l_p, o_a = carry
        v = entries_ref[t, :, :].astype(jnp.float32)  # [BLOCK_BS, D]

        # Inline RMSNorm
        rms = jnp.sqrt(jnp.mean(v ** 2, axis=-1, keepdims=True) + norm_eps)
        k = v / rms * scale[None, :]

        # Score
        score = jnp.sum(k * q[None, :], axis=-1)  # [BLOCK_BS]

        # Online softmax update
        m_new = jnp.maximum(m_p, score)
        correction = jnp.exp(m_p - m_new)
        p = jnp.exp(score - m_new)
        l_new = correction * l_p + p
        o_new = correction[:, None] * o_a + p[:, None] * v
        return m_new, l_new, o_new

    m_final, l_final, o_final = jax.lax.fori_loop(0, n_active, _body, (m_prev, l_prev, o_acc))

    out_ref[:] = (o_final / l_final[:, None]).astype(out_ref.dtype)


def depth_attention_pallas(
    entries: jnp.ndarray,
    query: jnp.ndarray,
    norm_scale: jnp.ndarray,
    norm_eps: float = 1e-5,
    n_active: int | None = None,
    block_bs: int = 512,
) -> jnp.ndarray:
    """Pallas-accelerated depth attention for TPU.

    Falls back to JAX implementation if Pallas is not available.
    """
    if not HAS_PALLAS:
        return depth_attention_jax(entries, query, norm_scale, norm_eps, n_active)

    max_T, B, S, D = entries.shape
    BS = B * S
    if n_active is None:
        n_active = max_T

    entries_flat = entries.reshape(max_T, BS, D)
    n_blocks = (BS + block_bs - 1) // block_bs
    # Pad BS to multiple of block_bs
    pad_bs = n_blocks * block_bs - BS
    if pad_bs > 0:
        entries_flat = jnp.pad(entries_flat, ((0, 0), (0, pad_bs), (0, 0)))

    kernel_fn = functools.partial(
        _depth_attn_pallas_kernel,
        n_active=n_active,
        D=D,
        norm_eps=norm_eps,
    )

    out_flat = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((n_blocks * block_bs, D), entries.dtype),
        grid=(n_blocks,),
        in_specs=[
            pl.BlockSpec((max_T, block_bs, D), lambda i: (0, i * block_bs, 0)),
            pl.BlockSpec((D,), lambda i: (0,)),
            pl.BlockSpec((D,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((block_bs, D), lambda i: (i * block_bs, 0)),
    )(entries_flat, norm_scale, query)

    # Remove padding
    out_flat = out_flat[:BS]
    return out_flat.reshape(B, S, D)


# ─── Flax Module ──────────────────────────────────────────────────────────────

class AttentionResidual(nn.Module):
    """Depth-wise attention over block outputs (arXiv:2603.15031).

    Zero-init queries for uniform initial attention (paper spec).
    Maintains a fixed-size buffer; entries written via index_update.

    Usage in the scan loop:
        # Before scan: initialize buffer
        ar_buf = jnp.zeros((max_blocks, B, S, d_model))
        ar_buf = ar_buf.at[0].set(prelude_output)
        ar_count = 1

        # Inside scan body:
        h_agg = attn_res(ar_buf, ar_count, block_idx=ar_count)
        ...  # core blocks produce h_new
        ar_buf = ar_buf.at[ar_count].set(h_new)
        ar_count += 1
    """
    d_model: int
    max_blocks: int
    norm_eps: float = 1e-5
    use_pallas: bool = False

    @nn.compact
    def __call__(
        self,
        entries: jnp.ndarray,
        n_active: int,
        block_idx: int = 0,
    ) -> jnp.ndarray:
        """Attend over accumulated block outputs.

        Args:
            entries:   [max_blocks, B, S, d] — buffer of block outputs
            n_active:  Number of valid entries
            block_idx: Current block position (selects pseudo-query)

        Returns:
            [B, S, d] — softmax-weighted aggregation
        """
        proj = self.param(
            "proj",
            nn.initializers.zeros,
            (self.max_blocks, self.d_model),
        )
        norm_scale = self.param(
            "norm_scale",
            nn.initializers.ones,
            (self.d_model,),
        )

        query = proj[jnp.minimum(block_idx, self.max_blocks - 1)]

        if self.use_pallas:
            return depth_attention_pallas(
                entries, query, norm_scale, self.norm_eps, n_active,
            )
        return depth_attention_jax(
            entries, query, norm_scale, self.norm_eps, n_active,
        )
