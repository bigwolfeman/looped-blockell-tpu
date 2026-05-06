"""Vanilla transformer with pluggable attention for subquadratic ablation.

d=768, 12 heads, 6 layers, d_ff=2048 (~117M params).
Attention module is swappable — register new implementations in ATTENTION_REGISTRY.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelConfig:
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 2048
    n_layers: int = 6
    vocab_size: int = 49152
    max_seq_len: int = 32768
    dropout: float = 0.0
    attn_type: str = "dense"
    attn_kwargs: dict = field(default_factory=dict)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_module: nn.Module):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = attn_module
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class SubQTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        from attention import create_attention
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, create_attention(cfg, layer_idx=i))
            for i in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SubQTransformer ({cfg.attn_type}): {n_params/1e6:.1f}M params")

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
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
