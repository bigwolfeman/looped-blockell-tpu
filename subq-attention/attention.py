"""Pluggable attention modules for subquadratic ablation.

Every attention module takes (x: [B, S, D]) and returns (out: [B, S, D]).
Causal masking is handled internally. RoPE is applied inside the module.

Register new attention types by adding to ATTENTION_REGISTRY.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ATTENTION_REGISTRY: dict[str, type] = {}

def _load_extension_modules():
    import importlib, pathlib
    for f in pathlib.Path(__file__).parent.glob("attention_*.py"):
        importlib.import_module(f.stem)


def register_attention(name: str):
    def decorator(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator


def create_attention(cfg, layer_idx: int = 0) -> nn.Module:
    cls = ATTENTION_REGISTRY.get(cfg.attn_type)
    if cls is None:
        raise ValueError(
            f"Unknown attention type '{cfg.attn_type}'. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}")
    return cls(cfg, layer_idx=layer_idx, **cfg.attn_kwargs)


class RotaryEmbedding(nn.Module):
    """RoPE — required for any attention method that uses positional info."""

    def __init__(self, d_head: int, max_seq_len: int = 32768, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        S = q.shape[2]
        cos = self.cos_cached[:S].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:S].unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class BaseAttention(nn.Module):
    """Base class for all attention modules."""

    def __init__(self, cfg, layer_idx: int = 0, **kwargs):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.layer_idx = layer_idx

        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head, cfg.max_seq_len)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, S, _ = x.shape
        q = self.W_q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k)
        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ─── Exp 0: Dense Attention ───────────────────────────────────────────────────

@register_attention("dense")
class DenseAttention(BaseAttention):
    """Standard O(n²) multi-head attention. Uses SDPA for efficiency."""

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)
        return self.W_o(out)


# ─── Exp 1: Sliding Window Attention ──────────────────────────────────────────

@register_attention("sliding_window")
class SlidingWindowAttention(BaseAttention):
    """O(n·w) sliding window attention. No long-range retrieval."""

    def __init__(self, cfg, layer_idx: int = 0, window_size: int = 4096, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        B, H, S, D = q.shape
        w = self.window_size

        # Build causal + banded mask: attend to at most w previous positions
        row_idx = torch.arange(S, device=x.device).unsqueeze(1)
        col_idx = torch.arange(S, device=x.device).unsqueeze(0)
        mask = (col_idx <= row_idx) & (col_idx >= row_idx - w + 1)
        attn_bias = torch.where(mask, 0.0, float("-inf"))
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=D ** -0.5
        )
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)


# ─── Exp 2: NSA (Native Sparse Attention) — Paper Faithful ───────────────────
# DeepSeek NSA: 3 parallel branches with SIGMOID gating.
# Branch 1: Compressed — learned MLP compresses blocks of KV, full attention over compressed
# Branch 2: Selected — REUSES compressed scores as block importance, top-n block gather
# Branch 3: Sliding window — local context
# Key: the compressed branch's attention scores ARE the selection mechanism.
# The O(n²/r) bottleneck is in the compressed branch, not a separate scoring step.

@register_attention("nsa")
class NSAAttention(BaseAttention):
    """DeepSeek NSA: 3-branch sparse attention with sigmoid gates.

    Compressed attention scores (O(n²/r)) are reused as block importance
    for the selection branch. This is the quadratic bottleneck.
    """

    def __init__(
        self, cfg, layer_idx: int = 0,
        compress_ratio: int = 32,
        n_select_blocks: int = 16,
        select_block_size: int = 64,
        window_size: int = 512,
        **kwargs,
    ):
        super().__init__(cfg, layer_idx=layer_idx)
        self.compress_ratio = compress_ratio
        self.n_select_blocks = n_select_blocks
        self.select_block_size = select_block_size
        self.window_size = window_size

        # Learned compression MLP (NOT mean pool — paper uses MLP with intra-block pos enc)
        self.k_compress = nn.Linear(self.d_head * compress_ratio, self.d_head, bias=False)
        self.v_compress = nn.Linear(self.d_head * compress_ratio, self.d_head, bias=False)

        # Sigmoid gate: independent per-head, per-token (NOT softmax — all branches can fire)
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 4, cfg.n_heads * 3, bias=False),
        )

    def _window_mask(self, S: int, w: int, device) -> Tensor:
        row = torch.arange(S, device=device).unsqueeze(1)
        col = torch.arange(S, device=device).unsqueeze(0)
        mask = (col <= row) & (col >= row - w + 1)
        return torch.where(mask, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q, k, v = self._project_qkv(x)
        H, D = self.n_heads, self.d_head
        r = self.compress_ratio

        # ─── Branch 1: Compressed attention (O(S²/r) — THE BOTTLENECK) ────
        n_cmp = S // r
        usable = n_cmp * r
        k_blocks = k[:, :, :usable].reshape(B, H, n_cmp, r * D)
        v_blocks = v[:, :, :usable].reshape(B, H, n_cmp, r * D)
        k_c = self.k_compress(k_blocks)  # [B, H, n_cmp, D]
        v_c = self.v_compress(v_blocks)

        # Causal mask: only attend to FULLY PAST compressed blocks.
        # Block j contains tokens j*r..(j+1)*r-1. A query at position i can
        # only see block j if (j+1)*r-1 < i, i.e. j < i//r (strict).
        block_pos = torch.arange(n_cmp, device=x.device)
        query_block = torch.arange(S, device=x.device) // r
        causal_cmp = block_pos.unsqueeze(0) < query_block.unsqueeze(1)
        bias_cmp = torch.where(causal_cmp, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

        scores_cmp = torch.matmul(q, k_c.transpose(-2, -1)) * (D ** -0.5) + bias_cmp
        # Prevent NaN gradients: queries with no valid blocks get uniform scores
        no_valid = (scores_cmp == float("-inf")).all(dim=-1, keepdim=True)
        scores_cmp = scores_cmp.masked_fill(no_valid, 0.0)
        attn_cmp = F.softmax(scores_cmp, dim=-1)  # [B, H, S, n_cmp]
        # Zero out attention for queries with no valid blocks (gate learns to skip)
        attn_cmp = attn_cmp.masked_fill(no_valid, 0.0)
        out_compress = torch.matmul(attn_cmp, v_c)

        # ─── Branch 2: Selected — reuse compressed scores as block importance ─
        # Sum attention weights across heads for block importance (GQA-style)
        block_importance = attn_cmp.sum(dim=1)  # [B, S, n_cmp]
        n_sel = min(self.n_select_blocks, n_cmp)
        _, top_block_idx = block_importance.topk(n_sel, dim=-1)  # [B, S, n_sel]

        # Gather full-resolution KV for selected blocks
        bs = self.select_block_size
        n_raw_blocks = S // bs
        usable_raw = n_raw_blocks * bs

        # Map compressed block indices → raw block indices
        # Each compressed block (size r) may span multiple selection blocks (size bs)
        ratio = max(1, r // bs)
        raw_indices = []
        for i in range(n_sel):
            base = top_block_idx[..., i:i+1] * ratio  # [B, S, 1]
            for offset in range(ratio):
                idx = (base + offset).clamp(max=n_raw_blocks - 1)
                raw_indices.append(idx)
        raw_indices = torch.cat(raw_indices, dim=-1)  # [B, S, n_sel*ratio]

        # Build attention mask from selected raw blocks
        attend_sel = torch.zeros(B, H, S, S, device=x.device, dtype=torch.bool)
        for i in range(raw_indices.shape[-1]):
            block_start = raw_indices[..., i:i+1] * bs  # [B, S, 1]
            for off in range(min(bs, 16)):
                pos = (block_start + off).clamp(max=S - 1)
                attend_sel.scatter_(3, pos.unsqueeze(1).expand(-1, H, -1, -1), True)

        causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        attend_sel = attend_sel & causal.unsqueeze(0).unsqueeze(0)

        scores_sel = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
        scores_sel = scores_sel.masked_fill(~attend_sel, float("-inf"))
        attn_sel = F.softmax(scores_sel, dim=-1)
        attn_sel = attn_sel.masked_fill(attn_sel.isnan(), 0.0)
        out_selected = torch.matmul(attn_sel, v)

        # ─── Branch 3: Sliding window (O(S·w)) ───────────────────────────
        out_window = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self._window_mask(S, self.window_size, x.device),
            scale=D ** -0.5,
        )

        # ─── Sigmoid gating (independent, not softmax) ───────────────────
        gates = torch.sigmoid(self.gate(x))  # [B, S, H*3]
        gates = gates.reshape(B, S, H, 3).permute(0, 2, 1, 3)  # [B, H, S, 3]

        out = (
            gates[..., 0:1] * out_compress
            + gates[..., 1:2] * out_selected
            + gates[..., 2:3] * out_window
        )
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)


# ─── Exp 3: NSA + LSH Router ─────────────────────────────────────────────────
# Replace the O(n²) selected branch with LSH bucketing.

@register_attention("nsa_lsh")
class NSALSHAttention(NSAAttention):
    """NSA with LSH replacing the quadratic selection branch.

    Hash Q and K using random projections, group into buckets,
    exact attention within same-bucket tokens. O(n · n/b · d).
    """

    def __init__(
        self, cfg, layer_idx: int = 0,
        compress_ratio: int = 32,
        select_k: int = 256,
        window_size: int = 512,
        n_buckets: int = 64,
        n_rounds: int = 4,
        **kwargs,
    ):
        super().__init__(
            cfg, layer_idx=layer_idx,
            compress_ratio=compress_ratio,
            select_k=select_k,
            window_size=window_size,
        )
        self.n_buckets = n_buckets
        self.n_rounds = n_rounds

        # Random projection matrices for hashing (not learned — SimHash)
        self.register_buffer(
            "hash_projections",
            torch.randn(n_rounds, self.d_head, n_buckets // 2),
            persistent=False,
        )

    def _lsh_hash(self, x: Tensor) -> Tensor:
        """Hash [B, H, S, D] → [B, H, n_rounds, S] bucket IDs."""
        B, H, S, D = x.shape
        hashes = []
        for r in range(self.n_rounds):
            h = torch.matmul(x, self.hash_projections[r])  # [B, H, S, n_buckets//2]
            h = (h > 0).int()
            powers = (2 ** torch.arange(h.shape[-1], device=h.device)).float()
            bucket_id = (h.float() @ powers).long() % self.n_buckets
            hashes.append(bucket_id)
        return torch.stack(hashes, dim=2)  # [B, H, n_rounds, S]

    def _branch_selected(self, q, k, v):
        """LSH-based selection replacing quadratic top-k.

        For each round: hash Q and K → same-bucket pairs attend.
        Union across rounds for better recall.
        """
        B, H, S, D = q.shape

        q_hashes = self._lsh_hash(q)  # [B, H, n_rounds, S]
        k_hashes = self._lsh_hash(k)

        # For each query, collect keys from same bucket across all rounds
        # Build attendance mask: attend if any round puts q,k in same bucket
        attend_mask = torch.zeros(B, H, S, S, device=q.device, dtype=torch.bool)
        for r in range(self.n_rounds):
            same_bucket = q_hashes[:, :, r].unsqueeze(-1) == k_hashes[:, :, r].unsqueeze(-2)
            attend_mask = attend_mask | same_bucket

        # Add causal constraint
        causal = torch.tril(torch.ones(S, S, device=q.device, dtype=torch.bool))
        attend_mask = attend_mask & causal.unsqueeze(0).unsqueeze(0)

        # Exact attention over attended positions
        scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
        scores = scores.masked_fill(~attend_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn.isnan(), 0.0)  # handle all-masked rows
        return torch.matmul(attn, v)


# ─── Exp 5: NSA + Learned Block Router ───────────────────────────────────────
# Our Block-ELL expertise applied to attention: score KV blocks, attend to top-k.

@register_attention("nsa_block_router")
class NSABlockRouterAttention(NSAAttention):
    """NSA with learned block-level routing for the selection branch.

    Divide KV into blocks of block_size tokens. Score each block using
    mean key embedding vs query. Select top-k blocks, exact attention
    within selected blocks. O(n/B · d) selection.
    """

    def __init__(
        self, cfg, layer_idx: int = 0,
        compress_ratio: int = 32,
        select_k: int = 256,
        window_size: int = 512,
        block_size: int = 128,
        top_blocks: int = 8,
        **kwargs,
    ):
        super().__init__(
            cfg, layer_idx=layer_idx,
            compress_ratio=compress_ratio,
            select_k=select_k,
            window_size=window_size,
        )
        self.block_size = block_size
        self.top_blocks = top_blocks

        # Learned scorer: takes (q, mean_k_block) → relevance score
        self.block_scorer = nn.Sequential(
            nn.Linear(self.d_head * 2, 64, bias=False),
            nn.SiLU(),
            nn.Linear(64, 1, bias=False),
        )

    def _branch_selected(self, q, k, v):
        """Block-level routing replacing quadratic top-k."""
        B, H, S, D = q.shape
        bs = self.block_size
        n_blocks = S // bs
        usable = n_blocks * bs
        top_b = min(self.top_blocks, n_blocks)

        # Mean-pool keys into block representations
        k_blocks = k[:, :, :usable].reshape(B, H, n_blocks, bs, D).mean(dim=3)  # [B,H,nB,D]

        # Score each (query, block) pair
        q_exp = q.unsqueeze(3).expand(-1, -1, -1, n_blocks, -1)  # [B,H,S,nB,D]
        k_exp = k_blocks.unsqueeze(2).expand(-1, -1, S, -1, -1)  # [B,H,S,nB,D]
        paired = torch.cat([q_exp, k_exp], dim=-1)  # [B,H,S,nB,2D]
        block_scores = self.block_scorer(paired).squeeze(-1)  # [B,H,S,nB]

        # Causal: can only attend to blocks that start before query position
        block_starts = torch.arange(n_blocks, device=q.device) * bs
        query_pos = torch.arange(S, device=q.device)
        causal = block_starts.unsqueeze(0) <= query_pos.unsqueeze(1)
        block_scores = block_scores.masked_fill(
            ~causal.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Select top blocks
        _, top_idx = block_scores.topk(top_b, dim=-1)  # [B,H,S,top_b]

        # Build attention mask from selected blocks
        attend_mask = torch.zeros(B, H, S, S, device=q.device, dtype=torch.bool)
        for bi in range(top_b):
            block_idx = top_idx[..., bi]  # [B,H,S]
            block_start = block_idx * bs  # [B,H,S]
            for offset in range(bs):
                key_pos = block_start + offset
                key_pos = key_pos.clamp(max=S - 1)
                # Mark these positions as attended
                attend_mask.scatter_(
                    3,
                    key_pos.unsqueeze(-1),
                    True,
                )

        # Causal constraint
        causal_full = torch.tril(torch.ones(S, S, device=q.device, dtype=torch.bool))
        attend_mask = attend_mask & causal_full.unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
        scores = scores.masked_fill(~attend_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn.isnan(), 0.0)
        return torch.matmul(attn, v)


# ─── Exp 6: Hierarchical LOD Attention ───────────────────────────────────────
# MegaAttention internalized: tree search over LOD hierarchy.

@register_attention("lod_attention")
class LODAttention(BaseAttention):
    """Hierarchical LOD attention — tree search from coarse to fine.

    LOD0 = raw KV tokens
    LOD1 = learned compression of 32 KV tokens → 1 summary
    LOD2 = learned compression of 32 LOD1 summaries → 1 summary

    Selection: Q attends to LOD2 → expand top LOD2 → attend LOD1 →
    expand top LOD1 → exact attention on LOD0 tokens.

    O(n · log(n)) via hierarchical pruning.
    """

    def __init__(
        self, cfg, layer_idx: int = 0,
        group_size: int = 32,
        top_lod2: int = 4,
        top_lod1: int = 8,
        window_size: int = 512,
        **kwargs,
    ):
        super().__init__(cfg, layer_idx=layer_idx)
        self.group_size = group_size
        self.top_lod2 = top_lod2
        self.top_lod1 = top_lod1
        self.window_size = window_size

        D = self.d_head
        # LOD1: compress group_size KVs → 1 summary
        self.k_lod1 = nn.Linear(D * group_size, D, bias=False)
        self.v_lod1 = nn.Linear(D * group_size, D, bias=False)
        # LOD2: compress group_size LOD1 summaries → 1 summary
        self.k_lod2 = nn.Linear(D * group_size, D, bias=False)
        self.v_lod2 = nn.Linear(D * group_size, D, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q, k, v = self._project_qkv(x)
        H, D = self.n_heads, self.d_head
        g = self.group_size

        # Build LOD hierarchy
        n_lod1 = S // g
        usable1 = n_lod1 * g

        k_groups1 = k[:, :, :usable1].reshape(B, H, n_lod1, g * D)
        v_groups1 = v[:, :, :usable1].reshape(B, H, n_lod1, g * D)
        k1 = self.k_lod1(k_groups1)  # [B, H, n_lod1, D]
        v1 = self.v_lod1(v_groups1)

        n_lod2 = n_lod1 // g
        if n_lod2 > 0:
            usable2 = n_lod2 * g
            k1_groups = k1[:, :, :usable2].reshape(B, H, n_lod2, g * D)
            v1_groups = v1[:, :, :usable2].reshape(B, H, n_lod2, g * D)
            k2 = self.k_lod2(k1_groups)  # [B, H, n_lod2, D]
            v2 = self.v_lod2(v1_groups)
        else:
            k2 = k1
            v2 = v1
            n_lod2 = n_lod1

        # Phase 1: Score against LOD2 (very coarse)
        scores_lod2 = torch.matmul(q, k2.transpose(-2, -1)) * (D ** -0.5)
        # Causal at LOD2 granularity
        lod2_pos = torch.arange(n_lod2, device=q.device) * g * g
        q_pos = torch.arange(S, device=q.device)
        # Strict: only score LOD2 blocks fully in the past
        lod2_end = lod2_pos + g * g - 1
        causal_lod2 = lod2_end.unsqueeze(0) < q_pos.unsqueeze(1)
        scores_lod2 = scores_lod2.masked_fill(
            ~causal_lod2.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        top2 = min(self.top_lod2, n_lod2)
        _, top2_idx = scores_lod2.topk(top2, dim=-1)  # [B, H, S, top2]

        # Phase 2: Expand top LOD2 → score LOD1 within those regions
        # Each LOD2 block covers g LOD1 blocks
        lod1_indices = []
        for i in range(top2):
            base = top2_idx[..., i] * g  # [B, H, S]
            for offset in range(g):
                idx = (base + offset).clamp(max=n_lod1 - 1)
                lod1_indices.append(idx)
        lod1_indices = torch.stack(lod1_indices, dim=-1)  # [B, H, S, top2*g]

        # Gather LOD1 keys for expanded regions
        lod1_exp = lod1_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        k1_gathered = torch.gather(
            k1.unsqueeze(2).expand(-1, -1, S, -1, -1), 3, lod1_exp
        )
        scores_lod1 = torch.matmul(q.unsqueeze(3), k1_gathered.transpose(-2, -1)).squeeze(3)
        scores_lod1 = scores_lod1 * (D ** -0.5)

        top1 = min(self.top_lod1, scores_lod1.shape[-1])
        _, top1_rel = scores_lod1.topk(top1, dim=-1)  # relative indices within gathered set

        # Map back to absolute LOD1 indices
        top1_abs = torch.gather(lod1_indices, -1, top1_rel)  # [B, H, S, top1]

        # Phase 3: Expand top LOD1 → exact LOD0 attention
        # Each LOD1 block covers g raw tokens
        attend_mask = torch.zeros(B, H, S, S, device=q.device, dtype=torch.bool)
        for i in range(top1):
            base = top1_abs[..., i] * g  # [B, H, S]
            for offset in range(g):
                pos = (base + offset).clamp(max=S - 1)
                attend_mask.scatter_(3, pos.unsqueeze(-1), True)

        # Add sliding window for local context
        w = self.window_size
        row_idx = torch.arange(S, device=q.device).unsqueeze(1)
        col_idx = torch.arange(S, device=q.device).unsqueeze(0)
        local_mask = (col_idx <= row_idx) & (col_idx >= row_idx - w + 1)
        attend_mask = attend_mask | local_mask.unsqueeze(0).unsqueeze(0)

        # Causal constraint
        causal = torch.tril(torch.ones(S, S, device=q.device, dtype=torch.bool))
        attend_mask = attend_mask & causal.unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
        scores = scores.masked_fill(~attend_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(attn.isnan(), 0.0)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)


# ─── Exp 7: MoSA — Mixture of Sparse Attention (NeurIPS 2025) ────────────────
# Each head selects its own k tokens via learned router BEFORE computing Q/K/V.
# Only selected tokens get projected → k×k attention instead of S×S.
# The ONLY sparse attention method that outperforms dense at equal compute.

@register_attention("mosa")
class MoSAAttention(BaseAttention):
    """MoSA: per-head learned token selection + exact sparse attention.

    Each head has a router sigmoid(X @ W_r) that scores every token.
    Top-k tokens per head get Q/K/V projections and participate in attention.
    Unselected tokens receive zero attention output.

    O(S·d_router + k²·d_head) per head. For k << S, this is subquadratic.
    """

    def __init__(
        self, cfg, layer_idx: int = 0,
        tokens_per_head: int = 512,
        window_size: int = 256,
        **kwargs,
    ):
        super().__init__(cfg, layer_idx=layer_idx)
        self.tokens_per_head = tokens_per_head
        self.window_size = window_size

        # Per-head router: scores each token for each head
        self.router = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        H, D = self.n_heads, self.d_head
        k = min(self.tokens_per_head, S)

        # Route: score every token for every head
        router_scores = torch.sigmoid(self.router(x))  # [B, S, H]
        router_scores = router_scores.permute(0, 2, 1)  # [B, H, S]

        # Select top-k tokens per head
        topk_scores, topk_idx = router_scores.topk(k, dim=-1)  # [B, H, k]

        # Full Q/K/V with RoPE (via base class)
        q_full, k_full, v_full = self._project_qkv(x)  # [B, H, S, D] each, with RoPE

        # Gather only selected positions for each head
        idx_d = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        q = torch.gather(q_full, 2, idx_d)       # [B, H, k, D]
        k_proj = torch.gather(k_full, 2, idx_d)
        v_proj = torch.gather(v_full, 2, idx_d)

        # Causal masking among selected tokens (by original position)
        pos_q = topk_idx.unsqueeze(-1)  # [B, H, k, 1]
        pos_k = topk_idx.unsqueeze(-2)  # [B, H, 1, k]
        causal = pos_k <= pos_q
        attn_bias = torch.where(causal, 0.0, float("-inf"))

        # Attention over selected tokens — k×k, not S×S
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) * (D ** -0.5)
        scores = scores + attn_bias
        attn = F.softmax(scores, dim=-1)
        out_sel = torch.matmul(attn, v_proj)  # [B, H, k, D]

        # Weight by router scores (differentiable path for routing gradients)
        out_sel = out_sel * topk_scores.unsqueeze(-1)  # [B, H, k, D]

        # Scatter back to full sequence
        out = torch.zeros(B, H, S, D, device=x.device, dtype=out_sel.dtype)
        out.scatter_(2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, D), out_sel)

        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)


# ─── Exp 8: NSA + MoSA Router (Synthesis) ────────────────────────────────────
# Our best hypothesis for what SubQ is doing: NSA's 3-branch architecture but
# with the compressed branch's quadratic scoring replaced by a MoSA-style
# learned router for the selection branch.

@register_attention("nsa_mosa")
class NSAMoSAAttention(BaseAttention):
    """NSA with MoSA-style learned router replacing quadratic selection.

    Branch 1: Compressed (learned MLP, for global context — still present)
    Branch 2: MoSA router selects top-k tokens → exact attention
    Branch 3: Sliding window
    Sigmoid gating combines all three.

    The compressed branch is CHEAP at this scale (S/r compressed tokens).
    The MoSA router replaces the quadratic selection with O(S) routing.
    """

    def __init__(
        self, cfg, layer_idx: int = 0,
        compress_ratio: int = 32,
        tokens_per_head: int = 512,
        window_size: int = 512,
        **kwargs,
    ):
        super().__init__(cfg, layer_idx=layer_idx)
        self.compress_ratio = compress_ratio
        self.tokens_per_head = tokens_per_head
        self.window_size = window_size

        self.k_compress = nn.Linear(self.d_head * compress_ratio, self.d_head, bias=False)
        self.v_compress = nn.Linear(self.d_head * compress_ratio, self.d_head, bias=False)

        # MoSA router for selection branch
        self.router = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)

        # Sigmoid gate (3 branches)
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 4, cfg.n_heads * 3, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q, k, v = self._project_qkv(x)
        H, D = self.n_heads, self.d_head
        r = self.compress_ratio

        # ─── Branch 1: Compressed (same as NSA, but NOT used for selection) ─
        n_cmp = S // r
        usable = n_cmp * r
        k_blocks = k[:, :, :usable].reshape(B, H, n_cmp, r * D)
        v_blocks = v[:, :, :usable].reshape(B, H, n_cmp, r * D)
        k_c = self.k_compress(k_blocks)
        v_c = self.v_compress(v_blocks)

        block_pos = torch.arange(n_cmp, device=x.device)
        query_block = torch.arange(S, device=x.device) // r
        causal_cmp = block_pos.unsqueeze(0) < query_block.unsqueeze(1)
        bias_cmp = torch.where(causal_cmp, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

        scores_cmp = torch.matmul(q, k_c.transpose(-2, -1)) * (D ** -0.5) + bias_cmp
        no_valid = (scores_cmp == float("-inf")).all(dim=-1, keepdim=True)
        scores_cmp = scores_cmp.masked_fill(no_valid, 0.0)
        attn_cmp = F.softmax(scores_cmp, dim=-1)
        attn_cmp = attn_cmp.masked_fill(no_valid, 0.0)
        out_compress = torch.matmul(attn_cmp, v_c)

        # ─── Branch 2: MoSA router-selected attention (O(S) routing) ─────
        tk = min(self.tokens_per_head, S)
        router_scores = torch.sigmoid(self.router(x)).permute(0, 2, 1)  # [B, H, S]
        topk_scores, topk_idx = router_scores.topk(tk, dim=-1)  # [B, H, tk]

        # Gather selected K, V
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        k_sel = torch.gather(k, 2, idx_exp)  # [B, H, tk, D]
        v_sel = torch.gather(v, 2, idx_exp)

        # Causal mask among selected positions
        pos_q = torch.arange(S, device=x.device).unsqueeze(0).unsqueeze(0)  # [1, 1, S]
        pos_k = topk_idx  # [B, H, tk]
        causal_sel = pos_k.unsqueeze(2) <= pos_q.unsqueeze(-1)  # [B, H, S, tk]

        scores_sel = torch.matmul(q, k_sel.transpose(-2, -1)) * (D ** -0.5)
        scores_sel = scores_sel.masked_fill(~causal_sel, float("-inf"))
        attn_sel = F.softmax(scores_sel, dim=-1)
        attn_sel = attn_sel.masked_fill(attn_sel.isnan(), 0.0)
        out_selected = torch.matmul(attn_sel, v_sel)

        # ─── Branch 3: Sliding window ────────────────────────────────────
        w = self.window_size
        row = torch.arange(S, device=x.device).unsqueeze(1)
        col = torch.arange(S, device=x.device).unsqueeze(0)
        win_mask = (col <= row) & (col >= row - w + 1)
        win_bias = torch.where(win_mask, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
        out_window = F.scaled_dot_product_attention(q, k, v, attn_mask=win_bias, scale=D**-0.5)

        # ─── Sigmoid gating ──────────────────────────────────────────────
        gates = torch.sigmoid(self.gate(x)).reshape(B, S, H, 3).permute(0, 2, 1, 3)
        out = (
            gates[..., 0:1] * out_compress
            + gates[..., 1:2] * out_selected
            + gates[..., 2:3] * out_window
        )
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)

# Load extension attention modules (attention_*.py)
_load_extension_modules()
