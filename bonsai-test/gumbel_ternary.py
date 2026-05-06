"""Gumbel-Softmax and Straight-Through Gumbel for ternary training.

Gumbel-Softmax (soft mode):
  Forward: soft ternary weight via reparameterized Gumbel-Softmax
  Backward: exact gradient through the relaxation (no estimator bias)
  Effective weight is continuous in [-1, +1] during training

Straight-Through Gumbel (hard mode, self.hard=True):
  Forward: hard ternary {-1, 0, +1} via argmax
  Backward: gradient through the soft Gumbel-Softmax (straight-through trick)
  Best of both: exact ternary forward + principled Gumbel gradient backward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from discrete_ternary_base import init_logits_from_ternary, apply_scales_and_mask, build_full_logits


class GumbelTernaryLinear(nn.Module):

    def __init__(self, out_features: int, in_features: int, initial_ternary: Tensor,
                 initial_scales: Tensor, weight_mask: Tensor, group_size: int = 128):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size

        logits = init_logits_from_ternary(initial_ternary, weight_mask)
        self.logits = nn.Parameter(logits)
        self.register_buffer("scales", initial_scales.clone().to(torch.bfloat16))
        self.register_buffer("mask", weight_mask.clone())

        self.tau = 1.0
        self.hard = False

    def forward(self, x: Tensor) -> Tensor:
        full = build_full_logits(self.logits, x.device)

        if self.training:
            gumbels = -torch.empty_like(full).exponential_().clamp(min=1e-10).log()
            y = F.softmax((full + gumbels) / self.tau, dim=-1)
        else:
            y = F.softmax(full / 0.01, dim=-1)

        if self.hard:
            idx = y.argmax(dim=-1)
            y_hard = F.one_hot(idx, 3).to(y.dtype)
            y = y_hard - y.detach() + y

        w = y[..., 2] - y[..., 0]
        w = apply_scales_and_mask(
            w, self.scales, self.mask, self.out_features, self.in_features, self.group_size)
        return F.linear(x, w.to(x.dtype))
