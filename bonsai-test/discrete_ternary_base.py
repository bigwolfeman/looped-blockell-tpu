"""Shared infrastructure for discrete ternary optimization methods.

All methods share:
- Per-weight logits [θ_neg, θ_pos] parameterizing P({-1, 0, +1})
- Frozen mask, scales, embed/lm_head/norms from Phase 1
- Same block architecture (attention + SwiGLU MLP)
- Shape-match-and-pop pattern for pruned data (fragile at same-shape layers
  but works when construction order matches pruning order)
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bitlinear import ternary_quantize


def init_logits_from_ternary(w_ternary: Tensor, mask: Tensor, peak: float = 2.0) -> Tensor:
    """Initialize 2-logit parameterization from existing ternary weights.

    Returns [out, in, 2]: (neg_logit, pos_logit). Zero logit is implicit 0.
    At peak=2.0: softmax([2,0,-2])≈[0.67, 0.24, 0.09] → ~67% on correct category.
    """
    w = w_ternary.float()
    neg = torch.where(w == -1, peak, -peak)
    pos = torch.where(w == 1, peak, -peak)
    logits = torch.stack([neg, pos], dim=-1)
    logits[~mask.bool()] = 0
    return logits


def apply_scales_and_mask(w: Tensor, scales: Tensor, mask: Tensor,
                          out_features: int, in_features: int, group_size: int) -> Tensor:
    """Apply per-group scales and binary mask to weight tensor [out, in]."""
    n_groups = out_features * in_features // group_size
    s = scales[:n_groups].float().unsqueeze(1).expand(n_groups, group_size)
    return w * s.reshape(out_features, in_features) * mask


def build_full_logits(logits_2: Tensor, device) -> Tensor:
    """Expand 2-logit [*, 2] to 3-category [*, 3] with zero in middle."""
    full = torch.zeros(*logits_2.shape[:-1], 3, device=device, dtype=logits_2.dtype)
    full[..., 0] = logits_2[..., 0]
    full[..., 2] = logits_2[..., 1]
    return full


class DiscreteTernaryBlock(nn.Module):
    """Transformer block with pluggable discrete ternary linear layers."""

    def __init__(self, cfg, base_attn, base_mlp, base_norms, pruned_layers, device, build_fn):
        super().__init__()
        self.norm_attn = base_norms[0]
        self.norm_mlp = base_norms[1]
        for p in self.norm_attn.parameters():
            p.requires_grad = False
        for p in self.norm_mlp.parameters():
            p.requires_grad = False

        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.W_q = build_fn(base_attn.W_q, pruned_layers, device)
        self.W_k = build_fn(base_attn.W_k, pruned_layers, device)
        self.W_v = build_fn(base_attn.W_v, pruned_layers, device)
        self.W_o = build_fn(base_attn.W_o, pruned_layers, device)

        if hasattr(base_attn, "alpha") and base_attn.alpha is not None:
            self.register_buffer("alpha", base_attn.alpha.detach().clone())
        else:
            self.alpha = None

        self.w_gate = build_fn(base_mlp.w_gate, pruned_layers, device)
        self.w_up = build_fn(base_mlp.w_up, pruned_layers, device)
        self.w_down = build_fn(base_mlp.w_down, pruned_layers, device)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        normed = self.norm_attn(x)
        q = self.W_q(normed).reshape(B, S, H, Dh).transpose(1, 2)
        k = self.W_k(normed).reshape(B, S, H, Dh).transpose(1, 2)
        v = self.W_v(normed).reshape(B, S, H, Dh).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if self.alpha is not None:
            x_heads = normed.reshape(B, S, H, Dh).transpose(1, 2)
            attn_out = attn_out + self.alpha * x_heads

        attn_out = self.W_o(attn_out.transpose(1, 2).reshape(B, S, D))
        x = x + attn_out

        normed = self.norm_mlp(x)
        x = x + self.w_down(F.silu(self.w_gate(normed)) * self.w_up(normed))
        return x


def _build_discrete_linear(bitlinear, pruned_layers, device, linear_cls, **kwargs):
    """Create discrete ternary linear from BitLinear + pruned data."""
    w = bitlinear.weight.detach()
    gs = bitlinear.group_size
    shape = tuple(w.shape)

    mask = torch.ones(shape, device=device)
    w_q, scales = ternary_quantize(w, gs)

    for name in list(pruned_layers.keys()):
        if pruned_layers[name]["shape"] == shape:
            data = pruned_layers.pop(name)
            w_q = data["ternary"]
            scales = data["scales"]
            mask = data["mask"]
            break

    return linear_cls(
        out_features=shape[0], in_features=shape[1],
        initial_ternary=w_q.to(device), initial_scales=scales.to(device),
        weight_mask=mask.to(device), group_size=gs, **kwargs,
    )


def create_discrete_model(p1_model, pruned_data, cfg, device, linear_cls, **linear_kwargs):
    """Build discrete ternary model from Phase 1 model + pruning data."""
    pd = dict(pruned_data)

    def build_fn(bitlinear, pl, dev):
        return _build_discrete_linear(bitlinear, pl, dev, linear_cls, **linear_kwargs)

    model = nn.Module()
    model.cfg = cfg

    model.embed = p1_model.embed
    model.final_norm = p1_model.final_norm
    model.lm_head = p1_model.lm_head
    for p in model.embed.parameters():
        p.requires_grad = False
    for p in model.final_norm.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False

    model.layers = nn.ModuleList()
    for layer in p1_model.layers:
        block = DiscreteTernaryBlock(
            cfg, base_attn=layer.attention, base_mlp=layer.mlp,
            base_norms=(layer.norm_attn, layer.norm_mlp),
            pruned_layers=pd, device=device, build_fn=build_fn,
        )
        model.layers.append(block)

    model.discrete_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, linear_cls):
            model.discrete_modules.append((name, mod))

    n_params = sum(p.numel() for _, m in model.discrete_modules for p in m.parameters())
    n_mods = len(model.discrete_modules)
    print(f"{linear_cls.__name__} model: {n_mods} modules, {n_params / 1e3:.1f}k learnable params")

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), labels.view(-1), ignore_index=-100)
        return out

    model.forward = types.MethodType(forward, model)
    return model.to(device)
