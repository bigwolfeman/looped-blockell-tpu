// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// swiglu_bwd.cuh — Fused Block-ELL SwiGLU backward pass declarations.
//
// Computes gradients for all three Block-ELL sparse projections (gate, up,
// down) in the SwiGLU MLP in a single backward call.
//
// SwiGLU forward recap (what we are differentiating through):
//
//   x         [N, d_model]   input and residual
//   gate      = BlockELL_gate(x)         [N, d_ff]   sparse matmul
//   up        = BlockELL_up(x)           [N, d_ff]   sparse matmul
//   h         = silu(gate) * up          [N, d_ff]   elementwise gate  ← SAVED
//   out       = BlockELL_down(h)         [N, d_model] sparse matmul
//   result    = out + x                  [N, d_model] residual
//
// Backward data flow given grad_output [N, d_model] (d(loss)/d(result)):
//
//   Step 1: residual — d_out = grad_output  (passthrough; accumulated into d_x later)
//   Step 2: down backward — d_h [N,d_ff], d_W_down [R_model,K_d,16,16]
//   Step 3: SiLU*up backward — d_gate, d_up (needs saved gate_pre_act, up_output)
//   Step 4: gate/up backward — d_x [N,d_model], d_W_gate, d_W_up
//           d_x also accumulates grad_output for the residual branch
//
// Block-ELL format reminder:
//   values:      [R, K, 16, 16]  bf16
//   col_indices: [R, K]          int32
//   Sparse matmul: y[r*16:(r+1)*16] += sum_k x[col[r,k]*16:(col[r,k]+1)*16] @ values[r,k]
//
// Kernel split (five launches for clarity and to keep register pressure low):
//   Kernel 1a: block_ell_down_backward_dh
//              grid (R_model, ceil(N/BLOCK_N)) — scatter d_h via atomicAdd
//   Kernel 1b: block_ell_down_backward_dw
//              grid (R_model*K_d,) — d_W_down via fp32 outer-product reduction
//   Kernel 2:  silu_mul_backward_kernel
//              grid (ceil(N*d_ff/256),) — elementwise d_gate, d_up
//   Kernel 3a: block_ell_proj_backward_dx
//              grid (R_ff*(K_g+K_u), ceil(N/BLOCK_N)) — scatter d_x via atomicAdd
//   Kernel 3b: residual_add_kernel
//              grid (ceil(N*d_model/256),) — d_x += grad_output
//   Kernel 3c: block_ell_proj_backward_dw
//              grid (R_ff*(K_g+K_u),) — d_W_gate, d_W_up via fp32 outer-product
//
// Accumulation precision:
//   - Weight gradients (d_W_*): fp32 accumulators, cast to bf16 on return
//   - Input gradients (d_h, d_x): bf16 atomicAdd (supported on sm_90a+)
//   - WMMA: __nv_bfloat16 A/B fragments, float C/D fragments
//
// Namespace: titan::block_ell
// Target:    sm_90a (safe superset; tested on RTX 5090 / sm_120)
//
// Branch: 006-looped-block-ell

#pragma once

#include <torch/types.h>

namespace titan {
namespace block_ell {

// ---------------------------------------------------------------------------
// SwiGLUBackwardResult
//
// Bundles all four output tensors from the backward pass.
//   d_x           [N, d_model] bf16  — gradient w.r.t. input x
//   d_values_gate [R_ff,   K_g, 16, 16] bf16 — gradient w.r.t. gate weights
//   d_values_up   [R_ff,   K_u, 16, 16] bf16 — gradient w.r.t. up weights
//   d_values_down [R_model, K_d, 16, 16] bf16 — gradient w.r.t. down weights
// ---------------------------------------------------------------------------
struct SwiGLUBackwardResult {
    torch::Tensor d_x;            // [N, d_model] bf16
    torch::Tensor d_values_gate;  // [R_ff,    K_g, 16, 16] bf16
    torch::Tensor d_values_up;    // [R_ff,    K_u, 16, 16] bf16
    torch::Tensor d_values_down;  // [R_model, K_d, 16, 16] bf16
};

// ---------------------------------------------------------------------------
// swiglu_backward_cuda
//
// Computes gradients for all learnable parameters and the input tensor of the
// fused SwiGLU Block-ELL MLP described in swiglu_fwd.cuh.
//
// All input tensors must reside on the same CUDA device.  Non-contiguous
// tensors are made contiguous internally.
//
// Arguments — saved from forward pass
// ------------------------------------
// grad_output   [N, d_model] bf16  — gradient of loss w.r.t. MLP output (result)
// x             [N, d_model] bf16  — original MLP input (also residual source)
// gate_pre_act  [N, d_ff]    bf16  — raw gate before SiLU (output of BlockELL_gate)
// up_output     [N, d_ff]    bf16  — up projection output (output of BlockELL_up)
// h             [N, d_ff]    bf16  — intermediate hidden: silu(gate_pre_act)*up_output
//                                    (returned by swiglu_forward_cuda_with_intermediate)
//
// Gate projection weights (Block-ELL)
// ------------------------------------
// values_gate   [R_ff, K_g, 16, 16] bf16
// col_idx_gate  [R_ff, K_g]         int32
//
// Up projection weights (Block-ELL)
// ----------------------------------
// values_up     [R_ff, K_u, 16, 16] bf16
// col_idx_up    [R_ff, K_u]         int32
//
// Down projection weights (Block-ELL)
// ------------------------------------
// values_down   [R_model, K_d, 16, 16] bf16
// col_idx_down  [R_model, K_d]         int32
//
// Returns
// -------
// SwiGLUBackwardResult containing {d_x, d_values_gate, d_values_up, d_values_down}.
//
// Throws
// ------
// c10::Error if shapes are inconsistent or tensors are not on CUDA.
// ---------------------------------------------------------------------------
SwiGLUBackwardResult swiglu_backward_cuda(
    torch::Tensor grad_output,   // [N, d_model] bf16
    torch::Tensor x,             // [N, d_model] bf16
    torch::Tensor gate_pre_act,  // [N, d_ff]    bf16  (pre-SiLU gate output)
    torch::Tensor up_output,     // [N, d_ff]    bf16  (up projection output)
    torch::Tensor h,             // [N, d_ff]    bf16  (silu(gate)*up, saved from fwd)
    // Gate projection
    torch::Tensor values_gate,   // [R_ff, K_g, 16, 16] bf16
    torch::Tensor col_idx_gate,  // [R_ff, K_g]         int32
    // Up projection
    torch::Tensor values_up,     // [R_ff, K_u, 16, 16] bf16
    torch::Tensor col_idx_up,    // [R_ff, K_u]         int32
    // Down projection
    torch::Tensor values_down,   // [R_model, K_d, 16, 16] bf16
    torch::Tensor col_idx_down   // [R_model, K_d]         int32
);

} // namespace block_ell
} // namespace titan
