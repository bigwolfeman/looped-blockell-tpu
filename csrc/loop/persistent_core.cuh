// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// persistent_core.cuh — Declarations for the persistent MLP core-loop step.
//
// "Persistent" here means the weight tensors are pinned in L2 cache via
// cudaAccessPolicyWindow (see l2_policy.cuh).  The kernel itself is a
// standard non-cooperative kernel; persistence comes from L2 residency,
// not from cooperative grid launch.
//
// Architecture context:
//   Each core-loop iteration contains:
//     1. Diagonal SSM injection (fuses h with injection input e)
//     2. For each core block i = 0..n_core-1:
//        a. RMSNorm (pre-MLP norm)
//        b. Block-ELL SwiGLU MLP (gate + up + SiLU + down + residual)
//     3. [Python] Scaled Dot-Product Attention between iterations
//
// This file handles steps 1 and 2 (MLP path).  Attention is called
// from Python using torch.nn.functional.scaled_dot_product_attention.
//
// Injection kernel math (diagonal SSM — discretized Mamba-style):
//
//   A  = -exp(log_A)                     (constrained negative diagonal)
//   dt = softplus(log_dt)                (constrained positive step size)
//   h  = exp(dt * A) * h  +  dt * e
//      = exp(-dt * exp(log_A)) * h  +  softplus(log_dt) * e
//
// This is a simple Hadamard multiply — no recurrence across the sequence
// dimension — because we treat each token independently in the loop
// (the SSM recurrence over T loop iterations, not over S tokens).
//
// RMSNorm kernel math:
//   h_normed[i] = h[i] * scale[i] / sqrt( mean(h^2) + eps )
//
// After injection + RMSNorm, control is handed to swiglu_forward_cuda
// from block_ell/swiglu_fwd.cuh for the actual sparse MLP computation.
//
// Namespace: titan::loop
// Target:    sm_90a (compatible with sm_120)
//
// Branch: 006-looped-block-ell

#pragma once

#include <vector>

#include <ATen/ATen.h>
#include <torch/types.h>

// Block-ELL SwiGLU forward — from Phase 2 kernels in csrc/block_ell/
// We call it inside persistent_mlp_step.
#include "../block_ell/swiglu_fwd.cuh"

namespace titan {
namespace loop {

// ---------------------------------------------------------------------------
// diagonal_injection_cuda
//
// Applies the diagonal SSM injection step in-place on `h`:
//
//   A  = -exp(log_A)                      [d_model]  fp32
//   dt = softplus(log_dt)                 [d_model]  fp32
//   h  = exp(dt * A) * h  +  dt * e
//
// This is a pure elementwise op across (N, d_model).
// Writes result into `h` in-place and also returns it for convenience.
//
// Arguments:
//   h       [N, d_model] bf16  — current hidden state; MUTATED in-place
//   e       [N, d_model] bf16  — injection input (constant across loop iters)
//   log_A   [d_model]    fp32  — log of |A| (A is negative; stored as -exp(log_A))
//   log_dt  [d_model]    fp32  — log of dt before softplus
//
// Returns h (same storage, for chaining).
// ---------------------------------------------------------------------------
at::Tensor diagonal_injection_cuda(
    at::Tensor       h,
    const at::Tensor& e,
    const at::Tensor& log_A,
    const at::Tensor& log_dt
);

// ---------------------------------------------------------------------------
// rmsnorm_cuda
//
// Applies RMSNorm with learned scale to each row of x:
//
//   rms    = sqrt( mean(x^2, dim=-1, keepdim=True) + eps )
//   output = x * scale / rms
//
// Arguments:
//   x      [N, d_model] bf16
//   scale  [d_model]    bf16  — per-feature learned scale (gamma)
//   eps    float              — numerical stability (default 1e-6)
//
// Returns output [N, d_model] bf16 (new tensor, x unchanged).
// ---------------------------------------------------------------------------
at::Tensor rmsnorm_cuda(
    const at::Tensor& x,
    const at::Tensor& scale,
    float             eps = 1e-6f
);

// ---------------------------------------------------------------------------
// persistent_mlp_step
//
// Runs one full core-loop MLP iteration: injection + (RMSNorm + Block-ELL
// SwiGLU) × n_core.  Returns the updated hidden state h.
//
// Call sequence per loop iteration from Python:
//   h = persistent_mlp_step(h, e, log_A, log_dt, ...)   # injection + MLPs
//   h = SDPA(h, ...)                                      # attention (Python)
//   # repeat T times
//
// Weight tensors should be pinned in L2 before the first iteration via
// titan::loop::pin_weights_l2().  They are READ-ONLY inside this function.
//
// Arguments:
//   h               [B*S, d_model]  bf16  — current hidden state (after SDPA)
//   e               [B*S, d_model]  bf16  — injection input (constant)
//   log_A           [d_model]       fp32  — SSM A parameter (log scale)
//   log_dt          [d_model]       fp32  — SSM dt parameter (log scale)
//   norm_scales     n_core × [d_model] bf16  — RMSNorm scale per core block
//   gate_values     n_core × [R_ff, K, 16, 16] bf16
//   gate_col_idx    n_core × [R_ff, K]          int32
//   up_values       n_core × [R_ff, K, 16, 16] bf16
//   up_col_idx      n_core × [R_ff, K]          int32
//   down_values     n_core × [R_model, K, 16, 16] bf16
//   down_col_idx    n_core × [R_model, K]          int32
//
// Returns h [B*S, d_model] bf16  (new tensor; input h is not modified).
// ---------------------------------------------------------------------------
at::Tensor persistent_mlp_step(
    at::Tensor               h,
    const at::Tensor&        e,
    const at::Tensor&        log_A,
    const at::Tensor&        log_dt,
    std::vector<at::Tensor>  norm_scales,
    std::vector<at::Tensor>  gate_values,
    std::vector<at::Tensor>  gate_col_idx,
    std::vector<at::Tensor>  up_values,
    std::vector<at::Tensor>  up_col_idx,
    std::vector<at::Tensor>  down_values,
    std::vector<at::Tensor>  down_col_idx
);

} // namespace loop
} // namespace titan
