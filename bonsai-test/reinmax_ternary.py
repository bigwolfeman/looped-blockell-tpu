"""REINMAX: 2nd-order gradient estimation for discrete ternary training.

Uses Heun's method (predictor-corrector) to get a second-order accurate
gradient estimate for categorical sampling. Lower bias AND lower variance
than standard STE or Gumbel-Softmax.

Forward: hard ternary sample (exact {-1, 0, +1})
Backward: Heun's method gradient at midpoint between soft probs and hard sample

Paper: arXiv:2304.08612, NeurIPS 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from discrete_ternary_base import init_logits_from_ternary, apply_scales_and_mask, build_full_logits


class TernaryReinMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, tau):
        y_soft = logits.float().softmax(dim=-1)
        gumbels = -torch.empty_like(logits).exponential_().clamp(min=1e-10).log()
        samples = (logits.float() + gumbels).argmax(dim=-1)
        one_hot = F.one_hot(samples, 3).to(y_soft.dtype)

        ctx.save_for_backward(one_hot, logits, y_soft)
        ctx.tau = tau
        return one_hot, y_soft

    @staticmethod
    def backward(ctx, g_D, g_p):
        one_hot, logits, y_soft = ctx.saved_tensors
        tau = ctx.tau

        if g_p is None:
            g_p = torch.zeros_like(g_D)

        pi1 = (logits.float() / tau).softmax(dim=-1)
        pi_mid = 0.5 * (pi1 + one_hot)

        raw1 = 2 * g_D * pi_mid
        grad1 = raw1 - pi_mid * raw1.sum(dim=-1, keepdim=True)

        raw0 = (-0.5 * g_D + g_p) * y_soft
        grad0 = raw0 - y_soft * raw0.sum(dim=-1, keepdim=True)

        grad = grad0 + grad1
        grad = grad - grad.mean(dim=-1, keepdim=True)
        return grad, None


class ReinmaxTernaryLinear(nn.Module):

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

    def forward(self, x: Tensor) -> Tensor:
        full = build_full_logits(self.logits, x.device)

        if self.training:
            one_hot, _ = TernaryReinMax.apply(full, self.tau)
            w = one_hot[..., 2] - one_hot[..., 0]
        else:
            p = F.softmax(full.float() / 0.01, dim=-1)
            w = p[..., 2] - p[..., 0]

        w = apply_scales_and_mask(
            w, self.scales, self.mask, self.out_features, self.in_features, self.group_size)
        return F.linear(x, w.to(x.dtype))
