"""NSA + Cosine Similarity Router — MegaContext internalized.

Replaces NSA's quadratic selection with cosine similarity between
query and mean-pooled KV block embeddings. No learned router params —
uses raw key geometry, exactly like MegaContext does externally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from attention import BaseAttention, register_attention


@register_attention("nsa_cosine")
class NSACosineAttention(BaseAttention):
    """NSA with cosine-similarity block selection.

    Branch 1: Compressed (learned MLP, global context)
    Branch 2: Cosine-selected (parameter-free block retrieval)
    Branch 3: Sliding window
    """

    def __init__(
        self, cfg, layer_idx: int = 0,
        compress_ratio: int = 32,
        block_size: int = 64,
        top_blocks: int = 16,
        window_size: int = 512,
        **kwargs,
    ):
        super().__init__(cfg, layer_idx=layer_idx)
        self.compress_ratio = compress_ratio
        self.block_size = block_size
        self.top_blocks = top_blocks
        self.window_size = window_size

        self.k_compress = nn.Linear(self.d_head * compress_ratio, self.d_head, bias=False)
        self.v_compress = nn.Linear(self.d_head * compress_ratio, self.d_head, bias=False)

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
        bs = self.block_size

        # ─── Branch 1: Compressed attention ──────────────────────────────
        n_cmp = S // r
        usable = n_cmp * r
        k_blocks_cmp = k[:, :, :usable].reshape(B, H, n_cmp, r * D)
        v_blocks_cmp = v[:, :, :usable].reshape(B, H, n_cmp, r * D)
        k_c = self.k_compress(k_blocks_cmp)
        v_c = self.v_compress(v_blocks_cmp)

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

        # ─── Branch 2: Cosine similarity block selection ─────────────────
        # Global per-head selection (like MegaContext): use mean query to
        # find relevant blocks, then all queries attend to selected blocks.
        n_blocks = S // bs
        top_b = min(self.top_blocks, n_blocks)

        # Mean-pool keys into block representatives
        k_mean = k[:, :, :n_blocks * bs].reshape(B, H, n_blocks, bs, D).mean(dim=3)
        k_mean = F.normalize(k_mean, dim=-1)  # [B, H, n_blocks, D]

        # Use mean query as retrieval key (like MegaContext uses last-256)
        q_mean = F.normalize(q.mean(dim=2, keepdim=True), dim=-1)  # [B, H, 1, D]
        cos_scores = torch.matmul(q_mean, k_mean.transpose(-2, -1)).squeeze(2)  # [B, H, n_blocks]

        # Select top-k blocks globally per head
        _, top_idx = cos_scores.topk(top_b, dim=-1)  # [B, H, top_b]

        # Expand to token indices
        token_indices = []
        for i in range(top_b):
            base = top_idx[..., i:i+1] * bs  # [B, H, 1]
            offsets = torch.arange(bs, device=x.device)
            token_indices.append(base + offsets)  # [B, H, bs]
        token_indices = torch.cat(token_indices, dim=-1).clamp(max=S - 1)  # [B, H, top_b*bs]
        n_sel = token_indices.shape[-1]

        # Gather selected K, V — [B, H, n_sel, D]
        idx_d = token_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        k_sel = torch.gather(k, 2, idx_d)
        v_sel = torch.gather(v, 2, idx_d)

        # Causal: each query can only attend to selected tokens at positions <= its own
        q_pos = torch.arange(S, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        sel_pos = token_indices.unsqueeze(2)  # [B, H, 1, n_sel]
        causal_sel = sel_pos <= q_pos  # [B, H, S, n_sel]

        # Attention over selected tokens — S×n_sel, NOT S×S
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
