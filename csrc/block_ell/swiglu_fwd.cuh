// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// swiglu_fwd.cuh — Fused Block-ELL SwiGLU forward pass declarations.
//
// Fuses three Block-ELL sparse matmuls + SiLU activation + gate multiply +
// residual add into two kernel launches, eliminating intermediate DRAM round-
// trips for the intermediate hidden state.
//
// SwiGLU MLP data flow:
//
//   x         [N, d_model]   (input, also residual if add_residual=true)
//      ↓
//   ┌─────────────── Phase 1 kernel ───────────────────────────────────────┐
//   │  gate = BlockELL_gate(x)   → [N, d_ff]  (86 block rows, K_g tiles)  │
//   │  up   = BlockELL_up(x)     → [N, d_ff]  (86 block rows, K_u tiles)  │
//   │  h    = silu(gate) * up    → [N, d_ff]  (elementwise, in registers)  │
//   └──────────────────────────────────────────────────────────────────────┘
//      ↓  intermediate h written to DRAM once
//   ┌─────────────── Phase 2 kernel ───────────────────────────────────────┐
//   │  out  = BlockELL_down(h)   → [N, d_model] (32 block rows, K_d tiles)│
//   │  (optionally) out += x     → residual add in epilogue                │
//   └──────────────────────────────────────────────────────────────────────┘
//      ↓
//   result    [N, d_model]
//
// Block-ELL format reminder:
//   values:      [R, K, 16, 16]  bf16  (R block rows, K alive tiles per row)
//   col_indices: [R, K]          int32 (which input column-block each tile maps to)
//   Sparse matmul:  y[..., r*16:(r+1)*16] += x[..., col*16:(col+1)*16] @ values[r,k]
//
// Dimensions for default 512→1376→512 SwiGLU at 25% density:
//   d_model = 512 → R_model = 32 block rows
//   d_ff    = 1376 → R_ff   = 86 block rows
//   K_g ≈ K_u ≈  8  (25% of 32 possible input column blocks)
//   K_d ≈ 22         (25% of 86 possible input column blocks)
//
// Key fusion savings:
//   - Gate and up projections share the same input x; shared memory is loaded
//     once per column-block and reused for both weight tiles when col_indices
//     happen to match (common after structured pruning keeps symmetric sparsity
//     patterns).  When they diverge the tiles are loaded independently.
//   - SiLU + gate multiply happens in registers before writing h, avoiding a
//     separate read-modify-write pass over d_ff elements.
//   - Residual add is folded into the Phase 2 store epilogue.
//
// Accumulation is always in fp32; final stores are bf16.
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
// swiglu_forward_cuda
//
// Computes:  result = BlockELL_down( silu(BlockELL_gate(x)) * BlockELL_up(x) )
//                   + (add_residual ? x : 0)
//
// All input tensors must reside on the same CUDA device and be contiguous
// (or will be made contiguous internally).
//
// Arguments
// ---------
// x             [N, d_model] bf16  — flattened token sequence (N = B*S)
//                                    also used as the skip-connection residual
//
// values_gate   [R_ff,   K_g, 16, 16] bf16   — gate projection weights
// col_idx_gate  [R_ff,   K_g]         int32  — input column-block indices (gate)
//
// values_up     [R_ff,   K_u, 16, 16] bf16   — up projection weights
// col_idx_up    [R_ff,   K_u]         int32  — input column-block indices (up)
//
// values_down   [R_model, K_d, 16, 16] bf16  — down projection weights
// col_idx_down  [R_model, K_d]         int32 — input column-block indices (down)
//
// add_residual  bool  — if true, add x to down-projection output in epilogue
//
// Returns
// -------
// Tensor [N, d_model] bf16  — result of the fused SwiGLU MLP
//
// Throws
// ------
// c10::Error if shapes are inconsistent or tensors are not on CUDA.
// ---------------------------------------------------------------------------
torch::Tensor swiglu_forward_cuda(
    torch::Tensor x,
    torch::Tensor values_gate,
    torch::Tensor col_idx_gate,
    torch::Tensor values_up,
    torch::Tensor col_idx_up,
    torch::Tensor values_down,
    torch::Tensor col_idx_down,
    bool          add_residual
);

// ---------------------------------------------------------------------------
// SwiGLUForwardIntermediates
//
// Bundles all intermediate tensors saved by the forward pass for the backward.
//   result       [N, d_model] bf16  — final MLP output
//   h            [N, d_ff]    bf16  — silu(gate)*up (input to down projection)
//   gate_pre_act [N, d_ff]    bf16  — gate output BEFORE SiLU (needed by silu_bwd)
//   up_output    [N, d_ff]    bf16  — up projection output (needed by silu_bwd)
// ---------------------------------------------------------------------------
struct SwiGLUForwardIntermediates {
    torch::Tensor result;        // [N, d_model] bf16
    torch::Tensor h;             // [N, d_ff]    bf16  silu(gate)*up
    torch::Tensor gate_pre_act;  // [N, d_ff]    bf16  pre-SiLU gate
    torch::Tensor up_output;     // [N, d_ff]    bf16  up projection
};

// ---------------------------------------------------------------------------
// swiglu_forward_cuda_with_intermediate
//
// Same as swiglu_forward_cuda but also returns the intermediate hidden state h
// (silu(gate)*up), the pre-SiLU gate output, and the up projection output.
// All three intermediates are required by the backward pass:
//   - h            → d_W_down = h^T @ grad_out
//   - gate_pre_act → SiLU derivative in silu_mul_backward
//   - up_output    → d_gate computation in silu_mul_backward
//
// Returns SwiGLUForwardIntermediates {result, h, gate_pre_act, up_output}.
// ---------------------------------------------------------------------------
SwiGLUForwardIntermediates swiglu_forward_cuda_with_intermediate(
    torch::Tensor x,
    torch::Tensor values_gate,
    torch::Tensor col_idx_gate,
    torch::Tensor values_up,
    torch::Tensor col_idx_up,
    torch::Tensor values_down,
    torch::Tensor col_idx_down,
    bool          add_residual
);

} // namespace block_ell
} // namespace titan
