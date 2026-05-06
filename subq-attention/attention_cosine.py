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
        n_blocks = S // bs
        top_b = min(self.top_blocks, n_blocks)

        # Mean-pool keys into block representatives (like MegaContext)
        k_mean = k[:, :, :n_blocks * bs].reshape(B, H, n_blocks, bs, D).mean(dim=3)
        k_mean = F.normalize(k_mean, dim=-1)
        q_norm = F.normalize(q, dim=-1)

        # Cosine similarity: each query vs each block mean
        cos_scores = torch.matmul(q_norm, k_mean.transpose(-2, -1))

        # Causal: only score blocks fully in the past
        block_end = torch.arange(n_blocks, device=x.device) * bs + bs - 1
        q_pos = torch.arange(S, device=x.device)
        causal_block = block_end.unsqueeze(0) < q_pos.unsqueeze(1)
        cos_scores = cos_scores.masked_fill(
            ~causal_block.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Top-k block selection
        _, top_idx = cos_scores.topk(top_b, dim=-1)

        # Build attendance mask from selected blocks
        attend_mask = torch.zeros(B, H, S, S, device=x.device, dtype=torch.bool)
        for i in range(top_b):
            block_start = top_idx[..., i] * bs
            for off in range(bs):
                pos = (block_start + off).clamp(max=S - 1)
                attend_mask.scatter_(3, pos.unsqueeze(-1), True)

        causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        attend_mask = attend_mask & causal.unsqueeze(0).unsqueeze(0)

        scores_sel = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
        scores_sel = scores_sel.masked_fill(~attend_mask, float("-inf"))
        attn_sel = F.softmax(scores_sel, dim=-1)
        attn_sel = attn_sel.masked_fill(attn_sel.isnan(), 0.0)
        out_selected = torch.matmul(attn_sel, v)

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
