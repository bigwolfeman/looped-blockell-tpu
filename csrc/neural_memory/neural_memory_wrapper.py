"""
Fused CUDA neural memory operations.

Drop-in replacement for NeuralMemory.update() and NeuralMemory.retrieve()
that eliminates the torch.autograd.grad graph break by using hand-rolled
backward through the memory MLP.

The CUDA ops (registered via memory_ops.cpp / memory_kernels.cu) implement
the full Titans Eq. 13-14 update as a single opaque kernel call:

    S_t = eta * S_{t-1} - theta * grad_l          (Eq. 13)
    M_t = (1 - alpha) * M_{t-1} + S_t             (Eq. 14)

where M_t refers to the flat concatenation of all memory MLP weights.

Usage:
    import titan_kernels  # loads TORCH_LIBRARY registrations
    from csrc.neural_memory.neural_memory_wrapper import (
        fused_memory_update,
        fused_memory_retrieve,
        patch_neural_memory,
    )

    # Functional API (use in custom training loops):
    loss = fused_memory_update(keys, values, memory_mlp, momentum_S,
                               eta=0.95, theta=0.01, alpha=0.001)
    output = fused_memory_retrieve(query_norm, memory_mlp)

    # Monkey-patch an existing NeuralMemory instance (simplest integration):
    patch_neural_memory(model.memory)  # replaces .update() and .retrieve()
"""

from __future__ import annotations

import math
import types
from typing import Optional

import torch
import torch.nn as nn

# Imported at module level to avoid repeated import overhead and to make the
# dependency explicit.  titans_core must be importable (editable install or on
# sys.path) before this module is loaded.
import titans_core.memory.neural_memory as _nm_module  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_linear_layers(memory_mlp: nn.Module) -> tuple[list[nn.Linear], int]:
    """
    Walk the memory MLP's Sequential and collect Linear layers in order.

    DeepMemoryMLP builds its Sequential as:
        Linear, SiLU, Linear, SiLU, ..., Linear  (no activation on output)

    Returns:
        linears: list of nn.Linear modules in forward order
        n_linear: count of linear layers found
    """
    linears: list[nn.Linear] = []
    sequential: nn.Sequential = memory_mlp.mlp  # type: ignore[attr-defined]
    for module in sequential:
        if isinstance(module, nn.Linear):
            linears.append(module)
    return linears, len(linears)


def _assert_four_layers(n: int) -> None:
    """The fused CUDA ops are compiled for exactly 4-layer MLPs."""
    if n != 4:
        raise ValueError(
            f"fused_memory_update/retrieve require exactly 4 Linear layers in the "
            f"memory MLP (matching the CUDA kernel ABI).  Got {n} layers.\n"
            f"Either use n_memory_layers=4 in your config, or implement a Python "
            f"fallback for other depths."
        )


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------

def fused_memory_update(
    keys: torch.Tensor,
    values: torch.Tensor,
    memory_mlp: nn.Module,
    momentum: torch.Tensor,
    eta: float,
    theta: float,
    alpha: float,
) -> torch.Tensor:
    """
    Fused in-place update of the neural memory MLP weights.

    Implements Titans Eq. 13-14 as a single CUDA kernel call with no Python-
    level graph break.  Mutates memory_mlp's weight/bias tensors and momentum
    in-place, exactly as the original NeuralMemory.update() does.

    Args:
        keys:       [N, d_memory] bf16 — projected keys W_K(x)
        values:     [N, d_memory] bf16 — projected values W_V(x)
        memory_mlp: DeepMemoryMLP (or any nn.Module with .mlp Sequential of
                    4 × Linear layers interleaved with SiLU activations)
        momentum:   [n_params] fp32 flat momentum buffer (register_buffer)
        eta:        Momentum decay coefficient (typically 0.95)
        theta:      Memory learning rate (typically 0.01)
        alpha:      Forget/decay rate for weight smoothing (0.0001–0.003)

    Returns:
        loss: Scalar [1] fp32 — the associative loss ||M(k) - v||² before
              the weight update, matching what NeuralMemory.update() returns.

    Raises:
        RuntimeError: if titan_kernels has not been loaded (no CUDA dispatch).
        ValueError:   if the memory MLP does not have exactly 4 Linear layers.
    """
    linears, n = _extract_linear_layers(memory_mlp)
    _assert_four_layers(n)

    w0, w1, w2, w3 = linears[0].weight, linears[1].weight, linears[2].weight, linears[3].weight
    b0, b1, b2, b3 = linears[0].bias,   linears[1].bias,   linears[2].bias,   linears[3].bias

    return torch.ops.titan.memory_update(
        keys, values,
        w0, w1, w2, w3,
        b0, b1, b2, b3,
        momentum,
        eta, theta, alpha,
    )


def fused_memory_retrieve(
    query_norm: torch.Tensor,
    memory_mlp: nn.Module,
) -> torch.Tensor:
    """
    Fused read-only forward pass through the neural memory MLP.

    Implements M*(q) = MLP(q_norm) without stop-gradient bookkeeping —
    that is handled in the patched retrieve() wrapper.  This function is
    a plain MLP inference call exposed as a custom op so Dynamo can see
    through it as an opaque call (no graph break).

    Args:
        query_norm: [N, d_memory] bf16 — normalized query (input_norm_k(W_Q(x)))
        memory_mlp: DeepMemoryMLP with 4 Linear layers

    Returns:
        output: [N, d_memory] bf16 — memory retrieval result M*(q)

    Raises:
        ValueError: if the memory MLP does not have exactly 4 Linear layers.
    """
    linears, n = _extract_linear_layers(memory_mlp)
    _assert_four_layers(n)

    w0, w1, w2, w3 = linears[0].weight, linears[1].weight, linears[2].weight, linears[3].weight
    b0, b1, b2, b3 = linears[0].bias,   linears[1].bias,   linears[2].bias,   linears[3].bias

    return torch.ops.titan.memory_retrieve(
        query_norm,
        w0, w1, w2, w3,
        b0, b1, b2, b3,
    )


# ---------------------------------------------------------------------------
# Monkey-patch helper
# ---------------------------------------------------------------------------

def patch_neural_memory(neural_memory_module: nn.Module) -> None:
    """
    Monkey-patch an existing NeuralMemory instance to use fused CUDA ops.

    Replaces the .update() and .retrieve() bound methods on the instance so
    that:
      - update() calls fused_memory_update() instead of torch.autograd.grad()
      - retrieve() calls fused_memory_retrieve() instead of torch.no_grad MLP

    The @torch._dynamo.disable decorator is NOT applied to the replacement
    methods — that's the whole point.  Dynamo sees these as opaque custom ops
    with no Python graph break.

    Cheap Python-level logic that IS graph-break-safe is preserved:
      - _compute_surprise_alpha() (scalar math, no autograd)
      - W_K / W_V / W_Q projections (standard nn.Linear, compile-safe)
      - input_norm_k normalization (compile-safe)
      - Straight-through estimator for W_Q gradients in retrieve()

    Checkpoint compatibility:
        Weight shapes and parameter names are UNCHANGED.  Any checkpoint saved
        by the original implementation loads cleanly after patching.

    Args:
        neural_memory_module: A NeuralMemory instance (titans_core.memory.neural_memory)

    Example:
        import titan_kernels
        from csrc.neural_memory.neural_memory_wrapper import patch_neural_memory

        model = TitanMAC(config)
        for layer in model.layers:
            if hasattr(layer, 'memory'):
                patch_neural_memory(layer.memory)
    """
    mod = neural_memory_module  # shorthand

    # -----------------------------------------------------------------------
    # Patched update() — replaces @torch._dynamo.disable original
    # -----------------------------------------------------------------------

    def _fused_update(
        self,
        x: torch.Tensor,
        theta_t: Optional[float] = None,
        return_stats: bool = False,
        differentiable: bool = False,
    ) -> torch.Tensor:
        """
        Fused memory update using CUDA kernel — no graph break.

        Mirrors NeuralMemory.update() contract exactly.  differentiable=True
        falls back to the original MAML path (that case still needs autograd.grad
        and is not performance-critical).

        Steps:
            a. Project k = W_K(x), v = W_V(x)  — normal nn.Linear, compile-safe
            b. Compute surprise-modulated alpha  — scalar Python, no graph break
            c. Call fused_memory_update()        — opaque CUDA op, no graph break
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, d_model], got {x.shape}")

        theta = theta_t if theta_t is not None else self.theta

        # Fall back to original path for MAML-style differentiable updates.
        # This case is rare (ablation only) and the graph break is acceptable.
        if differentiable and x.requires_grad:
            return _original_update(self, x, theta_t=theta_t,
                                    return_stats=return_stats, differentiable=True)

        # a. Project to key/value space
        # Shape: [B, T, d_model] → [B*T, d_memory] (flatten batch×seq for kernel)
        B, T, _ = x.shape
        k = self.W_K(x).reshape(B * T, -1)   # [N, d_memory] bf16
        v = self.W_V(x).reshape(B * T, -1)   # [N, d_memory] bf16

        # b. Compute surprise-modulated alpha (cheap scalar Python, no graph break)
        # We need a loss estimate for _compute_surprise_alpha.
        # Use the current memory output as a proxy — cheap single forward pass.
        with torch.no_grad():
            predicted = _run_mlp_forward(self.memory_mlp, k)
            loss_proxy = torch.nn.functional.mse_loss(predicted, v)

        alpha_val, eta_val = self._compute_surprise_alpha(loss_proxy)

        # c. Fused CUDA update: mutates MLP weights + momentum, returns loss [1] fp32.
        # Squeeze to [] scalar so the return shape matches the original update()
        # which returns F.mse_loss() — a zero-dim tensor.  Callers in titanmac.py
        # do `total = ce_loss + memory_loss`; mismatched shapes would silently
        # broadcast to [1] and break scalar-loss assumptions.
        loss = fused_memory_update(
            k, v,
            self.memory_mlp,
            self.momentum_S,
            eta=eta_val,
            theta=theta,
            alpha=alpha_val,
        ).squeeze(0)  # [1] fp32  →  [] fp32

        # Update diagnostic stats (same as original, for get_memory_stats())
        self._last_update_stats_tensors = {
            "alpha_t": torch.tensor(alpha_val),
            "eta_t": torch.tensor(eta_val),
            # grad_norm/clipped not available from fused kernel — report -1 sentinel
            "grad_norm": torch.tensor(-1.0),
            "grad_clipped": False,
            "skipped": False,
            "differentiable": False,
        }

        if return_stats:
            return {
                "loss": loss,
                "alpha_t": alpha_val,
                "eta_t": eta_val,
                "grad_norm": -1.0,   # not tracked in fused path
                "grad_clipped": False,
                "skipped": False,
                "differentiable": False,
            }
        return loss

    # -----------------------------------------------------------------------
    # Patched retrieve() — no @torch._dynamo.disable, no stop-gradient context
    # -----------------------------------------------------------------------

    def _fused_retrieve(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused memory retrieve using CUDA kernel — no graph break.

        Mirrors NeuralMemory.retrieve() contract exactly.

        Steps:
            a. Project q = W_Q(x)              — normal nn.Linear, compile-safe
            b. Normalize: q_norm = input_norm_k(q) — compile-safe
            c. Call fused_memory_retrieve()    — opaque CUDA op (read-only)
            d. Straight-through for W_Q grads  — q - q.detach(), compile-safe
        """
        # a. Query projection — gradient flows to W_Q through here
        q = self.W_Q(x)  # [B, T, d_memory]

        # b. Input normalization (same as original retrieve)
        q_norm = self.input_norm_k(q)  # [B, T, d_memory]

        # c. Fused MLP forward (read-only, no mutation)
        B, T, D = q_norm.shape
        q_norm_flat = q_norm.reshape(B * T, D)
        memory_out_flat = fused_memory_retrieve(q_norm_flat, self.memory_mlp)
        memory_output = memory_out_flat.reshape(B, T, D)  # [B, T, d_memory]

        # d. Straight-through estimator: value = memory_output (stop-grad on MLP
        #    weights), gradient path to W_Q via q - q.detach()
        #    This is functionally identical to the original retrieve(), but
        #    replaces the `with torch.no_grad(): memory_mlp(...)` call that
        #    forced a stop-gradient context in Python.
        if q.requires_grad:
            output = memory_output + (q - q.detach())
        else:
            output = memory_output

        return output

    # -----------------------------------------------------------------------
    # Capture the original update() so differentiable=True can fall back
    # -----------------------------------------------------------------------

    # The original NeuralMemory.update is decorated with @torch._dynamo.disable;
    # calling it as an unbound function with explicit `self` preserves that
    # behaviour for the MAML path (differentiable=True).
    # _nm_module is imported at module level (top of file).
    _original_update = _nm_module.NeuralMemory.update

    # -----------------------------------------------------------------------
    # Helper: vanilla Python MLP forward (used for loss_proxy in update)
    # -----------------------------------------------------------------------

    def _run_mlp_forward(memory_mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Execute the memory MLP forward pass without touching CUDA kernel."""
        return memory_mlp(x)

    # -----------------------------------------------------------------------
    # Bind new methods to the instance (not the class — preserves other instances)
    # -----------------------------------------------------------------------

    mod.update = types.MethodType(_fused_update, mod)
    mod.retrieve = types.MethodType(_fused_retrieve, mod)


# ---------------------------------------------------------------------------
# Batch-patch utility
# ---------------------------------------------------------------------------

def patch_all_neural_memories(model: nn.Module) -> int:
    """
    Recursively find all NeuralMemory instances in a model and patch them.

    Args:
        model: Any nn.Module (e.g., TitanMAC)

    Returns:
        count: Number of NeuralMemory instances patched

    Example:
        import titan_kernels
        from csrc.neural_memory.neural_memory_wrapper import patch_all_neural_memories

        model = TitanMAC(config)
        n = patch_all_neural_memories(model)
        print(f"Patched {n} NeuralMemory modules — graph breaks eliminated")
    """
    # Import here to avoid circular imports at module load time
    from titans_core.memory.neural_memory import NeuralMemory  # type: ignore

    count = 0
    for module in model.modules():
        if isinstance(module, NeuralMemory):
            patch_neural_memory(module)
            count += 1
    return count
