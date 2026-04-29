"""DeepSeek Sparse Attention for JAX/Flax.

Ported from BLT-Jepa blt_jepa/layers.py (DSAttention + LightningIndexer).

Two-level hierarchy:
1. Block-level: mean-pool into blocks, score with lightweight indexer
2. Token-level: expand selected blocks, run full attention on subset

Falls back to standard causal attention when seq_len is short enough
that sparse indexing overhead isn't worth it.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


def _precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> jnp.ndarray:
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(half, dtype=jnp.float32) / half))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)


def _apply_rope(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    B, H, S, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = freqs[:S, :, 0][None, None]
    sin = freqs[:S, :, 1][None, None]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class LightningIndexer(nn.Module):
    """Lightweight scoring network for block-level KV selection.

    Computes: I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s)
    """
    dim: int
    n_indexer_heads: int = 4

    @nn.compact
    def __call__(self, q_input: jnp.ndarray, k_input: jnp.ndarray) -> jnp.ndarray:
        """(B, Lq, D), (B, Lk, D) → (B, Lq, Lk) relevance scores."""
        head_dim = self.dim // self.n_indexer_heads

        q = nn.Dense(self.n_indexer_heads * head_dim, use_bias=False, name="wq")(q_input)
        k = nn.Dense(self.n_indexer_heads * head_dim, use_bias=False, name="wk")(k_input)
        w = nn.Dense(self.n_indexer_heads, use_bias=False, name="wg")(q_input)

        B, Lq, _ = q_input.shape
        Lk = k_input.shape[1]
        q = q.reshape(B, Lq, self.n_indexer_heads, head_dim)
        k = k.reshape(B, Lk, self.n_indexer_heads, head_dim)

        qk = jnp.einsum("bqhd,bkhd->bqhk", q, k)
        qk = jax.nn.relu(qk)
        scores = jnp.einsum("bqhk,bqh->bqk", qk, w)
        return scores


class DeepSeekSparseAttention(nn.Module):
    """DeepSeek Sparse Attention with block-level indexing.

    At short sequences (L <= 2 * n_blocks_selected * block_size), falls
    back to standard causal attention. At longer sequences, selects top-k
    KV blocks per query block via the LightningIndexer.
    """
    d_model: int
    n_heads: int
    max_seq_len: int = 2048
    top_k: int = 256
    n_indexer_heads: int = 4
    block_size: int = 32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        B, L, D = x.shape
        head_dim = self.d_model // self.n_heads

        q = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wq")(x)
        k = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wk")(x)
        v = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wv")(x)

        q = q.reshape(B, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        freqs = _precompute_rope_freqs(head_dim, self.max_seq_len)
        q = _apply_rope(q.astype(jnp.float32), freqs).astype(jnp.bfloat16)
        k = _apply_rope(k.astype(jnp.float32), freqs).astype(jnp.bfloat16)

        n_blocks_to_select = max(1, self.top_k // self.block_size)
        threshold = n_blocks_to_select * self.block_size * 2

        if L <= threshold:
            return self._full_causal_attention(q, k, v, x, B, L, head_dim)

        return self._sparse_attention(q, k, v, x, B, L, head_dim, n_blocks_to_select)

    def _full_causal_attention(self, q, k, v, x, B, L, head_dim):
        scale = math.sqrt(head_dim)
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        causal_mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))[None, None]
        NEG_INF = jnp.finfo(jnp.bfloat16).min
        attn_logits = jnp.where(causal_mask, attn_logits, NEG_INF)
        attn_w = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        out = jnp.matmul(attn_w, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return nn.Dense(self.d_model, use_bias=False, dtype=jnp.bfloat16, name="wo")(out)

    def _sparse_attention(self, q, k, v, x, B, L, head_dim, n_blocks_to_select):
        bs = self.block_size
        n_blocks = L // bs
        L_trimmed = n_blocks * bs

        x_blocks = x[:, :L_trimmed].reshape(B, n_blocks, bs, -1).mean(axis=2)

        indexer = LightningIndexer(
            dim=self.d_model,
            n_indexer_heads=self.n_indexer_heads,
            name="indexer",
        )
        block_scores = jax.lax.stop_gradient(indexer(x_blocks, x_blocks))

        n_sel = min(n_blocks_to_select, n_blocks)
        _, top_block_idx = jax.lax.top_k(block_scores, n_sel)

        block_starts = top_block_idx * bs
        offsets = jnp.arange(bs)
        token_indices = (block_starts[..., None] + offsets[None, None, None, :])
        token_indices = token_indices.reshape(B, n_blocks, n_sel * bs)
        total_selected = n_sel * bs

        token_indices = jnp.broadcast_to(
            token_indices[:, :, None, :],
            (B, n_blocks, bs, total_selected),
        ).reshape(B, L_trimmed, total_selected)

        if L > L_trimmed:
            remainder = L - L_trimmed
            full_idx = jnp.broadcast_to(
                jnp.arange(L)[None, None, :],
                (B, remainder, L),
            )
            if L > total_selected:
                token_indices = jnp.pad(token_indices, ((0, 0), (0, 0), (0, L - total_selected)))
                token_indices = jnp.concatenate([token_indices, full_idx], axis=1)
            else:
                token_indices = jnp.concatenate(
                    [token_indices, full_idx[:, :, :total_selected]], axis=1
                )

        idx = jnp.clip(token_indices, 0, L - 1)
        kv_len = idx.shape[-1]

        # Gather selected K, V for each query position
        # k_full: (B, H, L, d) → gather along L dimension per query
        def _gather_kv(kv, indices):
            # kv: (B, H, L, d), indices: (B, L_out, kv_len)
            B, H, L_full, d = kv.shape
            L_out = indices.shape[1]
            idx_exp = jnp.broadcast_to(
                indices[:, None, :, :, None],
                (B, H, L_out, kv_len, d),
            )
            kv_exp = jnp.broadcast_to(
                kv[:, :, None, :, :],
                (B, H, L_out, L_full, d),
            )
            return jnp.take_along_axis(kv_exp, idx_exp, axis=3)

        k_sel = _gather_kv(k, idx)  # (B, H, L, kv_len, d)
        v_sel = _gather_kv(v, idx)

        q_exp = q[:, :, :, None, :]  # (B, H, L, 1, d)
        scores = jnp.matmul(q_exp, k_sel.transpose(0, 1, 2, 4, 3)) / math.sqrt(head_dim)
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        out = jnp.matmul(scores, v_sel).squeeze(3)  # (B, H, L, d)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return nn.Dense(self.d_model, use_bias=False, dtype=jnp.bfloat16, name="wo")(out)
