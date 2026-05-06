"""MLA: Multi-head Latent Attention (simplified DeepSeek V2).

Low-rank KV compression: x → c_kv (d_compress) → K, V.
During inference, only c_kv is cached (d_compress << n_heads * d_head).
During training, equivalent to factored KV projection with bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from attention import BaseAttention, register_attention


@register_attention("mla")
class MLAAttention(BaseAttention):
    """Low-rank KV compression through learned bottleneck.

    Standard: W_k ∈ [d_model, kv_dim], W_v ∈ [d_model, kv_dim]
    MLA:      W_dkv ∈ [d_model, d_c], W_uk ∈ [d_c, kv_dim], W_uv ∈ [d_c, kv_dim]

    With kv_rank=256, kv_dim=768: 3× KV cache compression.
    Param count is similar (factored projection ≈ original), but the
    bottleneck forces a shared latent representation across heads.
    """

    def __init__(self, cfg, layer_idx: int = 0, kv_rank: int = 256, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        self.kv_rank = kv_rank
        kv_dim = cfg.n_heads * self.d_head
        del self.W_k, self.W_v
        self.W_dkv = nn.Linear(cfg.d_model, kv_rank, bias=False)
        self.W_uk = nn.Linear(kv_rank, kv_dim, bias=False)
        self.W_uv = nn.Linear(kv_rank, kv_dim, bias=False)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, S, _ = x.shape
        q = self.W_q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        c_kv = self.W_dkv(x)
        k = self.W_uk(c_kv).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_uv(c_kv).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k)
        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)
        return self.W_o(out)
