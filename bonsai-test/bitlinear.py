"""BitLinear: the core ternary layer.

ALL weights are ternary {-1, 0, +1}. No exceptions, no escape hatches.
Shadow weights in bf16 for optimizer, quantized on-the-fly for forward.

Follows Bonsai philosophy: quantize everything. The network learns to be
accurate through depth and discrete attractor dynamics, not precision.

Implementation:
  - Forward: quantize shadow weight → ternary matmul (simulated via F.linear)
  - Backward: STE (straight-through estimator) — gradient passes through quantization
  - Group-wise scaling: one fp16 scale factor per group of 128 elements
  - RMSNorm before each BitLinear (stabilizes activation distribution for quantization)
  - Activation quantization: 8-bit per-token absmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.scale


def ternary_quantize(w: Tensor, group_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize weights to {-1, 0, +1} with per-group AbsMean scaling.

    Args:
        w: weight tensor, any shape (will be reshaped to [-1, group_size])
        group_size: elements per scaling group

    Returns:
        w_q: ternary weight (same shape as w), values in {-1, 0, +1}
        scales: per-group scale factors [n_groups]
    """
    orig_shape = w.shape
    flat = w.reshape(-1, group_size)

    # Per-group absolute mean (the scaling factor)
    scales = flat.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)

    # Quantize: round(w / scale) clipped to [-1, 1]
    normalized = flat / scales
    w_q = normalized.round().clamp(-1, 1)

    return w_q.reshape(orig_shape), scales.squeeze(1)


def activation_quantize(x: Tensor, bits: int = 8) -> Tensor:
    """Per-token absmax quantization of activations to `bits` precision.

    Quantize → dequantize in one step (simulated quantization for training).
    """
    Qb = 2 ** (bits - 1)
    # Per-token (last dim) absmax
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = Qb / absmax
    # Quantize and immediately dequantize (simulate quantization noise)
    x_q = (x * scale).round().clamp(-Qb, Qb - 1) / scale
    return x_q


class STE(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization.

    Forward: returns quantized weight (ternary * scale)
    Backward: passes gradient through as if quantization didn't happen
    """
    @staticmethod
    def forward(ctx, w: Tensor, group_size: int) -> Tensor:
        w_q, scales = ternary_quantize(w, group_size)
        # Dequantize: multiply back by scale for correct magnitude
        n_groups = w.numel() // group_size
        scales_expanded = scales.unsqueeze(1).expand(n_groups, group_size).reshape(w.shape)
        return w_q * scales_expanded

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # STE: pass gradient through unchanged
        return grad_output, None


class BitLinear(nn.Module):
    """Fully ternary linear layer. No fp16 escape hatches.

    The weight parameter is the shadow weight (bf16, receives optimizer updates).
    Forward pass quantizes to ternary on-the-fly.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Shadow weight — this is what the optimizer sees
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Ensure dimensions are group-aligned
        total_elements = out_features * in_features
        if total_elements % group_size != 0:
            # Pad group_size down to nearest divisor
            self.group_size = self._find_group_size(total_elements, group_size)

        # Pre-quantization norm (stabilizes activation distribution)
        self.input_norm = RMSNorm(in_features)

    @staticmethod
    def _find_group_size(total: int, target: int) -> int:
        """Find largest group size <= target that divides total."""
        for gs in range(target, 0, -1):
            if total % gs == 0:
                return gs
        return 1

    def forward(self, x: Tensor) -> Tensor:
        # Quantize weights via STE (ternary + group scales)
        # NO activation quantization during training — only weights are ternary.
        # NO input_norm here — the block-level RMSNorm handles normalization.
        # Activation quantization is inference-only optimization.
        w_q = STE.apply(self.weight, self.group_size)
        return F.linear(x, w_q, self.bias)


class BitEmbedding(nn.Module):
    """Ternary embedding layer. Bonsai-style: no precision escape hatches.

    Shadow embedding in bf16, quantized to ternary for lookup.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

        total = num_embeddings * embedding_dim
        if total % group_size != 0:
            self.group_size = BitLinear._find_group_size(total, group_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        # Quantize full embedding table, then lookup
        w_q = STE.apply(self.weight, self.group_size)
        return F.embedding(input_ids, w_q)
