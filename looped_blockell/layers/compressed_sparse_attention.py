"""Compressed Sparse Attention (CSA) — DeepSeek-V4 style.

Two-level: compress tokens via learned pooling, then sparse-select top-k
compressed entries per query via Lightning Indexer.

Differs from DSA (block-level indexing on raw tokens):
- CSA learns to compress tokens (softmax-gated pooling)
- Operates over compressed KV entries (fewer, richer representations)
- Includes sliding window for recent uncompressed tokens
- Attention sinks (learnable, allow total mass < 1)

Reference: DeepSeek-V4 Technical Report (Apr 24, 2026)
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


class KVCompressor(nn.Module):
    """Learned softmax-gated pooling to compress token sequences.

    Groups of `ratio` tokens are compressed into a single entry via
    learned attention weights (not mean pooling).
    """
    d_model: int
    ratio: int = 8
    stride: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, L, d] → compressed: [B, L_c, d] where L_c ≈ L/stride."""
        B, L, D = x.shape
        n_windows = max(1, (L - self.ratio) // self.stride + 1)
        L_used = (n_windows - 1) * self.stride + self.ratio

        # Gather windows: [B, n_windows, ratio, D]
        indices = jnp.arange(self.ratio)[None, :] + jnp.arange(n_windows)[:, None] * self.stride
        indices = jnp.clip(indices, 0, L - 1)
        windows = x[:, indices, :]  # [B, n_windows, ratio, D]

        # Learned attention weights per position in window
        gate_logits = self.param(
            "gate_logits",
            nn.initializers.normal(stddev=0.02),
            (self.ratio,),
        )
        gate_weights = jax.nn.softmax(gate_logits)  # [ratio]

        # Weighted sum across window positions
        compressed = jnp.einsum("bwrd,r->bwd", windows, gate_weights)  # [B, n_windows, D]
        return compressed


class CSALightningIndexer(nn.Module):
    """Scores compressed KV entries for per-query top-k selection."""
    d_model: int
    n_indexer_heads: int = 4

    @nn.compact
    def __call__(self, q: jnp.ndarray, k_compressed: jnp.ndarray) -> jnp.ndarray:
        """q: [B, Lq, D], k_compressed: [B, Lc, D] → scores: [B, Lq, Lc]."""
        head_dim = self.d_model // self.n_indexer_heads

        qi = nn.Dense(self.n_indexer_heads * head_dim, use_bias=False, name="wq")(q)
        ki = nn.Dense(self.n_indexer_heads * head_dim, use_bias=False, name="wk")(k_compressed)
        w = nn.Dense(self.n_indexer_heads, use_bias=False, name="wg")(q)

        B, Lq, _ = q.shape
        Lc = k_compressed.shape[1]
        qi = qi.reshape(B, Lq, self.n_indexer_heads, head_dim)
        ki = ki.reshape(B, Lc, self.n_indexer_heads, head_dim)

        qk = jnp.einsum("bqhd,bkhd->bqhk", qi, ki)
        qk = jax.nn.relu(qk)
        scores = jnp.einsum("bqhk,bqh->bqk", qk, w)
        return scores


class CompressedSparseAttention(nn.Module):
    """CSA: compress tokens → select top-k → attend + sliding window.

    Falls back to standard causal attention when sequence is short.
    """
    d_model: int
    n_heads: int
    max_seq_len: int = 2048
    compress_ratio: int = 8
    compress_stride: int = 4
    top_k: int = 256
    window_size: int = 128
    n_indexer_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, L, D = x.shape
        head_dim = self.d_model // self.n_heads

        # Q, K, V projections
        q = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wq")(x)
        k = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wk")(x)
        v = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wv")(x)

        q = q.reshape(B, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        freqs = _precompute_rope_freqs(head_dim, self.max_seq_len)
        q = _apply_rope(q.astype(jnp.float32), freqs).astype(jnp.bfloat16)
        k = _apply_rope(k.astype(jnp.float32), freqs).astype(jnp.bfloat16)

        # Short sequences: full causal attention
        n_compressed = max(1, (L - self.compress_ratio) // self.compress_stride + 1)
        if L <= self.window_size * 2 or n_compressed <= self.top_k:
            return self._full_causal(q, k, v, B, L, head_dim)

        # ── Compress KV ──
        compressor = KVCompressor(
            d_model=self.d_model,
            ratio=self.compress_ratio,
            stride=self.compress_stride,
            name="kv_compressor",
        )
        x_compressed = compressor(x)  # [B, Lc, D]
        Lc = x_compressed.shape[1]

        # Compressed K, V
        k_c = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wk_c")(x_compressed)
        v_c = nn.Dense(self.n_heads * head_dim, use_bias=False, dtype=jnp.bfloat16, name="wv_c")(x_compressed)
        k_c = k_c.reshape(B, Lc, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v_c = v_c.reshape(B, Lc, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        # ── Lightning Indexer: select top-k compressed entries per query ──
        indexer = CSALightningIndexer(
            d_model=self.d_model,
            n_indexer_heads=self.n_indexer_heads,
            name="indexer",
        )
        scores = jax.lax.stop_gradient(indexer(x, x_compressed))  # [B, L, Lc]
        n_select = min(self.top_k, Lc)
        _, top_idx = jax.lax.top_k(scores, n_select)  # [B, L, n_select]

        # Gather selected compressed K, V
        # k_c: [B, H, Lc, d] → gather per query → [B, H, L, n_select, d]
        top_idx_exp = jnp.broadcast_to(
            top_idx[:, None, :, :, None],
            (B, self.n_heads, L, n_select, head_dim),
        )
        k_c_exp = jnp.broadcast_to(k_c[:, :, None, :, :], (B, self.n_heads, L, Lc, head_dim))
        v_c_exp = jnp.broadcast_to(v_c[:, :, None, :, :], (B, self.n_heads, L, Lc, head_dim))
        k_sel = jnp.take_along_axis(k_c_exp, top_idx_exp, axis=3)
        v_sel = jnp.take_along_axis(v_c_exp, top_idx_exp, axis=3)

        # ── Sliding window: last window_size uncompressed tokens ──
        window_mask = jnp.abs(
            jnp.arange(L)[None, :] - jnp.arange(L)[:, None]
        ) < self.window_size
        window_mask = window_mask & jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))

        # ── Combined attention ──
        scale = math.sqrt(head_dim)

        # Compressed branch: q @ k_sel^T
        q_exp = q[:, :, :, None, :]  # [B, H, L, 1, d]
        compressed_scores = jnp.matmul(q_exp, k_sel.transpose(0, 1, 2, 4, 3)) / scale
        compressed_scores = compressed_scores.squeeze(3)  # [B, H, L, n_select]

        # Window branch: standard causal with window mask
        window_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        NEG_INF = jnp.finfo(jnp.bfloat16).min
        window_scores = jnp.where(window_mask[None, None], window_scores, NEG_INF)

        # Attention sink: learnable logit that absorbs "no relevant context"
        sink_logit = self.param("sink_logit", nn.initializers.zeros, (self.n_heads, 1))
        sink_broadcast = jnp.broadcast_to(
            sink_logit[None, :, None, :],
            (B, self.n_heads, L, 1),
        )

        # Concatenate all score sources: [compressed | window | sink]
        all_scores = jnp.concatenate([
            compressed_scores.astype(jnp.float32),
            window_scores.astype(jnp.float32),
            sink_broadcast.astype(jnp.float32),
        ], axis=-1)  # [B, H, L, n_select + L + 1]

        attn_w = jax.nn.softmax(all_scores, axis=-1).astype(jnp.bfloat16)

        # Split weights back
        w_compressed = attn_w[..., :n_select]
        w_window = attn_w[..., n_select:n_select + L]
        # sink weight is discarded (absorbed attention mass)

        # Weighted sum
        out_compressed = jnp.matmul(w_compressed[:, :, :, None, :], v_sel).squeeze(3)
        out_window = jnp.matmul(w_window, v)
        out = out_compressed + out_window

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return nn.Dense(self.d_model, use_bias=False, dtype=jnp.bfloat16, name="wo")(out)

    def _full_causal(self, q, k, v, B, L, head_dim):
        scale = math.sqrt(head_dim)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))[None, None]
        NEG_INF = jnp.finfo(jnp.bfloat16).min
        scores = jnp.where(mask, scores, NEG_INF)
        w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        out = jnp.matmul(w, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return nn.Dense(self.d_model, use_bias=False, dtype=jnp.bfloat16, name="wo")(out)
