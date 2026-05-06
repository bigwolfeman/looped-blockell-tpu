"""Full-ternary transformer with residual attention.

Everything is BitLinear. No mixed precision. Bonsai-style.
Residual attention provides stable gradient flow through ternary layers.

Architecture: standard causal LM transformer
  - BitEmbedding (ternary)
  - N × TransformerBlock:
      - RMSNorm → BitLinear Q/K/V → Attention (with residual) → BitLinear O
      - RMSNorm → BitLinear gate/up (SwiGLU) → BitLinear down
  - RMSNorm → BitLinear LM head (NOT weight-tied — ternary embed != ternary head)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import BitLinear, BitEmbedding, RMSNorm


@dataclass
class TernaryConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1376  # (8/3)*512 rounded to 128 for SwiGLU
    n_layers: int = 6
    vocab_size: int = 49152
    max_seq_len: int = 1024
    dropout: float = 0.0
    group_size: int = 128
    # Residual attention: out = attn(x) + alpha * x
    attn_residual: bool = True
    attn_residual_init: float = 0.1


class ResidualAttention(nn.Module):
    """Multi-head attention with learned residual connection.

    out = softmax(QK^T/sqrt(d)) V + alpha * x

    The residual weight alpha starts small (0.1) and is learned.
    Provides guaranteed gradient flow even when ternary attention
    weights are poorly calibrated early in training.
    """

    def __init__(self, cfg: TernaryConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = BitLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)
        self.W_k = BitLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)
        self.W_v = BitLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)
        self.W_o = BitLinear(cfg.d_model, cfg.d_model, group_size=cfg.group_size)

        if cfg.attn_residual:
            self.alpha = nn.Parameter(
                torch.full((cfg.n_heads, 1, 1), cfg.attn_residual_init)
            )
        else:
            self.alpha = None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.W_q(x).reshape(B, S, H, Dh).transpose(1, 2)
        k = self.W_k(x).reshape(B, S, H, Dh).transpose(1, 2)
        v = self.W_v(x).reshape(B, S, H, Dh).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)

        # Residual attention: blend attention output with input
        if self.alpha is not None:
            x_heads = x.reshape(B, S, H, Dh).transpose(1, 2)
            out = out + self.alpha * x_heads

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_o(out)


class SwiGLU_MLP(nn.Module):
    """SwiGLU MLP with all-ternary projections."""

    def __init__(self, cfg: TernaryConfig):
        super().__init__()
        self.w_gate = BitLinear(cfg.d_model, cfg.d_ff, group_size=cfg.group_size)
        self.w_up = BitLinear(cfg.d_model, cfg.d_ff, group_size=cfg.group_size)
        self.w_down = BitLinear(cfg.d_ff, cfg.d_model, group_size=cfg.group_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TernaryConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = ResidualAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU_MLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class TernaryTransformer(nn.Module):
    """Full-ternary causal language model.

    Every linear projection is BitLinear. Embeddings are BitEmbedding.
    The LM head is a separate BitLinear (not weight-tied) because ternary
    quantization of embed vs head may converge to different ternary patterns.
    """

    def __init__(self, cfg: TernaryConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = BitEmbedding(cfg.vocab_size, cfg.d_model, group_size=cfg.group_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = BitLinear(cfg.d_model, cfg.vocab_size, group_size=cfg.group_size)

        # Causal mask (precomputed)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TernaryTransformer: {n_params/1e6:.1f}M params (all ternary)")

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict:
        B, S = input_ids.shape
        x = self.embed(input_ids)

        mask = self.causal_mask[:, :, :S, :S]

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        return out
