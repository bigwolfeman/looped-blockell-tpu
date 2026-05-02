"""Multi-head causal attention with RoPE, GQA, QK-Norm, and merged CSA for JAX/Flax."""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from .norms import RMSNorm
from .compressed_sparse_attention import KVCompressor


def _precompute_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> jnp.ndarray:
    """Precompute RoPE frequency table [max_seq_len, head_dim // 2, 2].

    Returns complex-valued rotation factors as real pairs [cos, sin].
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # [T, half]
    return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)  # [T, half, 2]


def _apply_rope(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary position embedding to x [B, n_heads, S, head_dim].

    freqs: [S, head_dim // 2, 2]  (cos, sin pairs)
    """
    B, H, S, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]          # each [B, H, S, half]
    cos = freqs[:S, :, 0]                            # [S, half]
    sin = freqs[:S, :, 1]                            # [S, half]
    cos = cos[None, None, :, :]                      # [1, 1, S, half]
    sin = sin[None, None, :, :]
    rotated = jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
    )
    return rotated


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention.

    Supports:
    - GQA (n_kv_heads < n_heads): separate Q and KV projections, KV heads repeated
    - QK-Norm: RMSNorm on Q and K per head_dim, applied AFTER projection, BEFORE RoPE
    - CSA (compressed sparse attention): merged from CompressedSparseAttention, only
      activated when use_csa=True AND seq_len > csa_window_size
    - XSA (exclusive self attention, arXiv:2603.09078)
    """
    n_heads: int
    d_model: int
    max_seq_len: int = 1024
    dropout: float = 0.0
    use_xsa: bool = False
    dtype: jnp.dtype = jnp.bfloat16
    # GQA
    n_kv_heads: Optional[int] = None   # None → same as n_heads (full MHA)
    # QK-Norm
    use_qk_norm: bool = False
    # CSA fields (only used when use_csa=True)
    use_csa: bool = False
    csa_compress_ratio: int = 8
    csa_compress_stride: int = 4
    csa_window_size: int = 128
    csa_n_indexer_heads: int = 4

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: [B, S, d_model]
            deterministic: If False, apply dropout.

        Returns:
            [B, S, d_model]
        """
        B, S, D = x.shape
        head_dim = self.d_model // self.n_heads
        assert D == self.d_model

        n_kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        kv_groups = self.n_heads // n_kv

        # ── Q / K / V projections ──────────────────────────────────────────────
        if n_kv == self.n_heads:
            # Fused QKV (full MHA)
            qkv = nn.Dense(
                3 * self.d_model,
                use_bias=False,
                dtype=self.dtype,
                name="qkv_proj",
            )(x)  # [B, S, 3*D]
            q, k, v = jnp.split(qkv, 3, axis=-1)  # each [B, S, D]
        else:
            # Separate Q and KV projections (GQA)
            kv_dim = n_kv * head_dim
            q = nn.Dense(
                self.d_model,
                use_bias=False,
                dtype=self.dtype,
                name="q_proj",
            )(x)  # [B, S, D]
            kv = nn.Dense(
                2 * kv_dim,
                use_bias=False,
                dtype=self.dtype,
                name="kv_proj",
            )(x)  # [B, S, 2*kv_dim]
            k, v = jnp.split(kv, 2, axis=-1)  # each [B, S, kv_dim]

        # Reshape to [B, n_heads/n_kv, S, head_dim]
        q = q.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)  # [B, H, S, D]
        k = k.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)          # [B, Hkv, S, D]
        v = v.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)          # [B, Hkv, S, D]

        # ── QK-Norm (AFTER projection, BEFORE RoPE) ────────────────────────────
        if self.use_qk_norm:
            # Separate learned scale per (head_dim,) — not shared across heads
            q = RMSNorm(eps=1e-5, name="q_norm")(q)
            k = RMSNorm(eps=1e-5, name="k_norm")(k)

        # ── RoPE ──────────────────────────────────────────────────────────────
        freqs = _precompute_freqs(head_dim, self.max_seq_len)
        freqs = freqs.astype(jnp.float32)
        q = _apply_rope(q.astype(jnp.float32), freqs).astype(self.dtype)
        k = _apply_rope(k.astype(jnp.float32), freqs).astype(self.dtype)

        # ── CSA branch ────────────────────────────────────────────────────────
        if self.use_csa and S > self.csa_window_size:
            out = self._forward_csa(x, q, k, v, B, S, head_dim, n_kv, kv_groups, deterministic)
            out = out.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
            out = nn.Dense(
                self.d_model,
                use_bias=False,
                dtype=self.dtype,
                name="out_proj",
            )(out)
            return out

        # ── Standard causal attention ─────────────────────────────────────────
        # Expand KV heads to match Q heads (GQA)
        if kv_groups > 1:
            k = jnp.repeat(k, kv_groups, axis=1)  # [B, H, S, D]
            v = jnp.repeat(v, kv_groups, axis=1)

        # Causal mask
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))[None, None, :, :]
        additive_mask = jnp.where(
            mask,
            jnp.zeros((S, S), dtype=self.dtype),
            jnp.finfo(self.dtype).min,
        )

        scale = math.sqrt(head_dim)
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # [B, H, S, S]
        attn_logits = attn_logits + additive_mask
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(self.dtype)

        if self.dropout > 0.0:
            attn_weights = nn.Dropout(rate=self.dropout)(attn_weights, deterministic=deterministic)

        attn_out = jnp.matmul(attn_weights, v)  # [B, H, S, head_dim]

        # XSA: subtract projection onto own value vector (arXiv:2603.09078)
        if self.use_xsa:
            v_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True).clip(min=1e-8)
            proj = jnp.sum(attn_out * v, axis=-1, keepdims=True) / v_norm_sq
            attn_out = attn_out - proj * v

        # Merge heads
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)

        # Output projection
        out = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype,
            name="out_proj",
        )(attn_out)

        return out

    def _forward_csa(
        self,
        x: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        B: int,
        S: int,
        head_dim: int,
        n_kv: int,
        kv_groups: int,
        deterministic: bool,
    ) -> jnp.ndarray:
        """CSA: compress tokens → sliding window + compressed global + sink.

        q/k/v are already projected, QK-normed, and RoPE'd.
        k/v are in GQA dimensions [B, n_kv, S, head_dim].
        Returns attn_out [B, n_heads, S, head_dim] (pre-merged, pre-out_proj).
        """
        scale = math.sqrt(head_dim)
        W = self.csa_window_size
        n_heads = self.n_heads

        # Expand KV heads for window branch
        if kv_groups > 1:
            k_win = jnp.repeat(k, kv_groups, axis=1)  # [B, H, S, D]
            v_win = jnp.repeat(v, kv_groups, axis=1)
        else:
            k_win, v_win = k, v

        # ── Compress input for compressed KV branch ────────────────────────────
        compressor = KVCompressor(
            d_model=self.d_model,
            ratio=self.csa_compress_ratio,
            stride=self.csa_compress_stride,
            name="kv_compressor",
        )
        x_c = compressor(x)          # [B, Lc, D]
        Lc = x_c.shape[1]

        # Compressed K and V — projected using GQA dimensions (n_kv heads)
        kv_dim_c = n_kv * head_dim
        kv_c = nn.Dense(
            2 * kv_dim_c,
            use_bias=False,
            dtype=self.dtype,
            name="kv_proj_c",
        )(x_c)  # [B, Lc, 2*kv_dim_c]
        k_c, v_c = jnp.split(kv_c, 2, axis=-1)
        k_c = k_c.reshape(B, Lc, n_kv, head_dim).transpose(0, 2, 1, 3)  # [B, Hkv, Lc, D]
        v_c = v_c.reshape(B, Lc, n_kv, head_dim).transpose(0, 2, 1, 3)

        # Apply QK-Norm to compressed K if enabled
        if self.use_qk_norm:
            k_c = RMSNorm(eps=1e-5, name="k_norm")(k_c)

        # Expand compressed KV heads
        if kv_groups > 1:
            k_c = jnp.repeat(k_c, kv_groups, axis=1)  # [B, H, Lc, D]
            v_c = jnp.repeat(v_c, kv_groups, axis=1)

        # ── Compressed branch scores with causal masking ───────────────────────
        # entry_end[j] = j*stride + ratio - 1 → query i can attend if i >= entry_end[j]
        compressed_scores = jnp.matmul(q, k_c.transpose(0, 1, 3, 2)) / scale  # [B, H, S, Lc]
        stride = self.csa_compress_stride
        ratio = self.csa_compress_ratio
        entry_end = jnp.arange(Lc) * stride + ratio - 1  # [Lc]
        query_pos = jnp.arange(S)                          # [S]
        causal_c = query_pos[:, None] >= entry_end[None, :]  # [S, Lc]
        NEG_INF = jnp.finfo(jnp.bfloat16).min
        compressed_scores = jnp.where(
            causal_c[None, None, :, :],
            compressed_scores,
            NEG_INF,
        )

        # ── Sliding window scores ──────────────────────────────────────────────
        window_scores = jnp.matmul(q, k_win.transpose(0, 1, 3, 2)) / scale  # [B, H, S, S]
        pos = jnp.arange(S)
        dist = pos[:, None] - pos[None, :]  # [S, S], dist[i,j] = i - j
        window_mask = (dist >= 0) & (dist < W)
        window_scores = jnp.where(window_mask[None, None, :, :], window_scores, NEG_INF)

        # ── Attention sink ─────────────────────────────────────────────────────
        sink_logit = self.param(
            "sink_logit",
            nn.initializers.zeros,
            (n_heads, 1),
        )
        sink_broadcast = jnp.broadcast_to(
            sink_logit[None, :, None, :],
            (B, n_heads, S, 1),
        )

        # ── Combined softmax over [compressed | window | sink] ─────────────────
        all_scores = jnp.concatenate([
            compressed_scores.astype(jnp.float32),
            window_scores.astype(jnp.float32),
            sink_broadcast.astype(jnp.float32),
        ], axis=-1)  # [B, H, S, Lc + S + 1]

        attn_w = jax.nn.softmax(all_scores, axis=-1).astype(self.dtype)

        w_c = attn_w[..., :Lc]          # [B, H, S, Lc]
        w_w = attn_w[..., Lc:Lc + S]   # [B, H, S, S]
        # sink weight absorbed (discarded)

        out_c = jnp.matmul(w_c, v_c)   # [B, H, S, head_dim]
        out_w = jnp.matmul(w_w, v_win) # [B, H, S, head_dim]
        out = out_c + out_w             # [B, H, S, head_dim]

        # XSA applied before merge (uses expanded v_win)
        if self.use_xsa:
            v_norm_sq = jnp.sum(v_win * v_win, axis=-1, keepdims=True).clip(min=1e-8)
            proj = jnp.sum(out * v_win, axis=-1, keepdims=True) / v_norm_sq
            out = out - proj * v_win

        return out  # [B, H, S, head_dim]
