"""bf16 baseline transformer — same architecture, standard nn.Linear.

Identical to TernaryTransformer but with standard layers for A/B comparison.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import RMSNorm


@dataclass
class BaselineConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1376
    n_layers: int = 6
    vocab_size: int = 49152
    max_seq_len: int = 1024
    dropout: float = 0.0
    attn_residual: bool = True
    attn_residual_init: float = 0.1


class BaselineAttention(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

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

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        if self.alpha is not None:
            x_heads = x.reshape(B, S, H, Dh).transpose(1, 2)
            out = out + self.alpha * x_heads

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_o(out)


class BaselineMLP(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class BaselineBlock(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attention = BaselineAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = BaselineMLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.norm_attn(x), mask)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([BaselineBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"BaselineTransformer: {n_params/1e6:.1f}M params (bf16)")

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
