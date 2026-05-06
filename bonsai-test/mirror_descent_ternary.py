"""Mirror Descent on the probability simplex for ternary training.

The natural geometry for categorical distributions is the probability simplex
with KL divergence as the Bregman divergence. Standard Adam on softmax logits
computes gradients through the softmax Jacobian (mixing cross-terms). Mirror
Descent takes gradients in probability space directly, then updates logits.

Key difference from Gumbel-Softmax (Adam on logits via chain rule):
  Gumbel:  loss → ∂L/∂z via softmax Jacobian → Adam(z)
  MD:      loss → ∂L/∂p directly → Adam(z, grad=∂L/∂p)

The probability-space gradient avoids the Jacobian's cross-term mixing,
giving each category's gradient independent signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from discrete_ternary_base import init_logits_from_ternary, apply_scales_and_mask, build_full_logits


class MirrorDescentLinear(nn.Module):

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
        self._p_leaf = None

    def forward(self, x: Tensor) -> Tensor:
        full = build_full_logits(self.logits, x.device)
        p = F.softmax(full.float(), dim=-1)

        if self.training:
            p_leaf = p.detach().requires_grad_(True)
            self._p_leaf = p_leaf
        else:
            p_leaf = p

        w = p_leaf[..., 2] - p_leaf[..., 0]
        w = apply_scales_and_mask(
            w, self.scales, self.mask, self.out_features, self.in_features, self.group_size)
        return F.linear(x, w.to(x.dtype))

    def transfer_gradients(self):
        """Transfer prob-space gradient to logits for optimizer update.

        Must be called after backward() and before optimizer.step().
        """
        if self._p_leaf is not None and self._p_leaf.grad is not None:
            g = self._p_leaf.grad
            self.logits.grad = torch.stack([g[..., 0], g[..., 2]], dim=-1)
        self._p_leaf = None
