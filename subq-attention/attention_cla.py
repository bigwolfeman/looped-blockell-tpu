"""CLA: Cross-Layer Attention + Per-layer Mixing Gate.

CLA: Even layers compute fresh KV, odd layers reuse cached KV.
Mixing Gate: Learned per-head gate blending dense and MoSA per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from attention import BaseAttention, register_attention, _rotate_half

_CLA_KV_CACHE: dict[int, tuple[Tensor, Tensor]] = {}


@register_attention("cla")
class CLAAttention(BaseAttention):
    """Cross-Layer Attention: reuse KV from a source layer.

    Even layers (0, 2, 4) compute and cache fresh KV.
    Odd layers (1, 3, 5) reuse KV from the preceding even layer.
    Saves 50% of KV computation. Critical for looped architectures
    where loop iterations can share KV from iteration 0.

    Cache is cleared at layer 0 forward (layers must execute in order).
    Gradients flow through cached KV back to the source layer.
    """

    def __init__(self, cfg, layer_idx: int = 0, share_interval: int = 2, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        self.share_interval = share_interval
        self.is_source = (layer_idx % share_interval) == 0

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape

        if self.layer_idx == 0:
            _CLA_KV_CACHE.clear()

        if self.is_source:
            q, k, v = self._project_qkv(x)
            _CLA_KV_CACHE[self.layer_idx] = (k, v)
        else:
            q = self.W_q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
            cos = self.rope.cos_cached[:S].unsqueeze(0).unsqueeze(0)
            sin = self.rope.sin_cached[:S].unsqueeze(0).unsqueeze(0)
            q = (q * cos) + (_rotate_half(q) * sin)
            source = (self.layer_idx // self.share_interval) * self.share_interval
            k, v = _CLA_KV_CACHE[source]

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)


@register_attention("mixing_gate")
class MixingGateAttention(BaseAttention):
    """Learned per-head gate blending dense and MoSA attention.

    gate ∈ [0,1] per head: out = gate·dense + (1-gate)·mosa.
    Initialized at sigmoid(0) = 0.5. Each layer discovers its optimal
    attention pattern — some may prefer full dense, others MoSA.
    """

    def __init__(self, cfg, layer_idx: int = 0, tokens_per_head: int = 512, **kwargs):
        super().__init__(cfg, layer_idx=layer_idx)
        self.tokens_per_head = tokens_per_head
        self.router = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)
        self.mix_gate = nn.Parameter(torch.zeros(cfg.n_heads))

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._project_qkv(x)
        B, H, S, D = q.shape

        out_dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        tk = min(self.tokens_per_head, S)
        router_scores = torch.sigmoid(self.router(x)).permute(0, 2, 1)
        topk_scores, topk_idx = router_scores.topk(tk, dim=-1)

        idx_d = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        k_sel = torch.gather(k, 2, idx_d)
        v_sel = torch.gather(v, 2, idx_d)

        pos_q = torch.arange(S, device=x.device).unsqueeze(0).unsqueeze(0)
        causal_sel = topk_idx.unsqueeze(2) <= pos_q.unsqueeze(-1)

        scores_sel = torch.matmul(q, k_sel.transpose(-2, -1)) * (D ** -0.5)
        scores_sel = scores_sel.masked_fill(~causal_sel, float("-inf"))
        attn_sel = F.softmax(scores_sel, dim=-1)
        attn_sel = attn_sel.masked_fill(attn_sel.isnan(), 0.0)
        out_mosa = torch.matmul(attn_sel, v_sel)

        gate = torch.sigmoid(self.mix_gate).reshape(1, H, 1, 1)
        out = gate * out_dense + (1 - gate) * out_mosa

        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(out)
