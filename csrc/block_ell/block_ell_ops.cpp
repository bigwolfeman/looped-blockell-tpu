/**
 * block_ell_ops.cpp — torch.library registration for Block-ELL SwiGLU ops.
 *
 * Registers four custom ops under the `titan::` namespace:
 *
 *   titan::block_ell_swiglu_fwd(x, v_gate, ci_gate,
 *                                v_up, ci_up,
 *                                v_down, ci_down,
 *                                add_residual) -> Tensor
 *     Fused forward pass: gate + up → silu(gate)*up → down + optional residual.
 *     Returns result [N, d_model] bf16.
 *
 *   titan::block_ell_swiglu_fwd_with_intermediate(x, v_gate, ci_gate,
 *                                                  v_up, ci_up,
 *                                                  v_down, ci_down,
 *                                                  add_residual)
 *       -> (Tensor result, Tensor h, Tensor gate_pre_act, Tensor up_output)
 *     Same as _fwd but also returns three intermediates required by backward:
 *       h            [N, d_ff] bf16  — silu(gate)*up (needed for d_W_down)
 *       gate_pre_act [N, d_ff] bf16  — gate before SiLU (needed for SiLU derivative)
 *       up_output    [N, d_ff] bf16  — up projection output (needed for d_gate)
 *
 *   titan::block_ell_swiglu_bwd(grad_out, x, gate_pre_act, up_out, h,
 *                                v_gate, ci_gate,
 *                                v_up, ci_up,
 *                                v_down, ci_down)
 *       -> (Tensor dx, Tensor d_vg, Tensor d_vu, Tensor d_vd)
 *     Backward pass: returns d_x (bf16) and weight grads (bf16).
 *
 * Meta implementations for shape inference under torch.compile are included
 * so Dynamo can trace through the custom ops without touching CUDA.
 *
 * Build: add block_ell_ops.cpp, swiglu_fwd.cu, swiglu_bwd.cu to setup.py sources.
 */

#include <torch/library.h>
#include <torch/types.h>
#include <ATen/ATen.h>

// Forward declarations — implemented in swiglu_fwd.cu / swiglu_bwd.cu

#include "swiglu_fwd.cuh"
#include "swiglu_bwd.cuh"

// ---------------------------------------------------------------------------
// Schema registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(titan, m) {
    // ---- Forward (result only) ----
    m.def(
        "block_ell_swiglu_fwd("
        "    Tensor x,"                  // [N, d_model] bf16
        "    Tensor v_gate,"             // [R_ff,   K_g, 16, 16] bf16
        "    Tensor ci_gate,"            // [R_ff,   K_g]         int32
        "    Tensor v_up,"               // [R_ff,   K_u, 16, 16] bf16
        "    Tensor ci_up,"              // [R_ff,   K_u]         int32
        "    Tensor v_down,"             // [R_model, K_d, 16, 16] bf16
        "    Tensor ci_down,"            // [R_model, K_d]         int32
        "    bool add_residual"          // fold x into down-proj epilogue
        ") -> Tensor"                    // [N, d_model] bf16
    );

    // ---- Forward with intermediates (for autograd) ----
    m.def(
        "block_ell_swiglu_fwd_with_intermediate("
        "    Tensor x,"
        "    Tensor v_gate,"
        "    Tensor ci_gate,"
        "    Tensor v_up,"
        "    Tensor ci_up,"
        "    Tensor v_down,"
        "    Tensor ci_down,"
        "    bool add_residual"
        ") -> (Tensor, Tensor, Tensor, Tensor)"  // (result, h, gate_pre_act, up_output) all [N,d_ff/d_model] bf16
    );

    // ---- Backward ----
    // Saved inputs from forward: x, gate_pre_act, up_out, h
    //   h is the intermediate silu(gate)*up returned by _fwd_with_intermediate.
    //   It is needed to compute d_W_down = h^T @ grad_out.
    // Returns: (dx, d_v_gate, d_v_up, d_v_down) all bf16
    m.def(
        "block_ell_swiglu_bwd("
        "    Tensor grad_out,"           // [N, d_model] bf16
        "    Tensor x,"                  // [N, d_model] bf16 (saved)
        "    Tensor gate_pre_act,"       // [N, d_ff]    bf16 (saved pre-SiLU gate)
        "    Tensor up_out,"             // [N, d_ff]    bf16 (saved up output)
        "    Tensor h,"                  // [N, d_ff]    bf16 (saved intermediate: silu(gate)*up)
        "    Tensor v_gate,"             // [R_ff,   K_g, 16, 16] bf16
        "    Tensor ci_gate,"            // [R_ff,   K_g]         int32
        "    Tensor v_up,"               // [R_ff,   K_u, 16, 16] bf16
        "    Tensor ci_up,"              // [R_ff,   K_u]         int32
        "    Tensor v_down,"             // [R_model, K_d, 16, 16] bf16
        "    Tensor ci_down"             // [R_model, K_d]         int32
        ") -> (Tensor, Tensor, Tensor, Tensor)"  // (dx, d_vg, d_vu, d_vd) bf16
    );
}

// ---------------------------------------------------------------------------
// CUDA implementation dispatch
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(titan, CUDA, m) {
    m.impl("block_ell_swiglu_fwd",
        [](at::Tensor x,
           at::Tensor v_gate,
           at::Tensor ci_gate,
           at::Tensor v_up,
           at::Tensor ci_up,
           at::Tensor v_down,
           at::Tensor ci_down,
           bool add_residual) -> at::Tensor {
            return titan::block_ell::swiglu_forward_cuda(
                x,
                v_gate, ci_gate,
                v_up,   ci_up,
                v_down, ci_down,
                add_residual
            );
        }
    );

    m.impl("block_ell_swiglu_fwd_with_intermediate",
        [](at::Tensor x,
           at::Tensor v_gate,
           at::Tensor ci_gate,
           at::Tensor v_up,
           at::Tensor ci_up,
           at::Tensor v_down,
           at::Tensor ci_down,
           bool add_residual) -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
            auto fwd = titan::block_ell::swiglu_forward_cuda_with_intermediate(
                x,
                v_gate, ci_gate,
                v_up,   ci_up,
                v_down, ci_down,
                add_residual
            );
            return {fwd.result, fwd.h, fwd.gate_pre_act, fwd.up_output};
        }
    );

    m.impl("block_ell_swiglu_bwd",
        [](at::Tensor grad_out,
           at::Tensor x,
           at::Tensor gate_pre_act,
           at::Tensor up_out,
           at::Tensor h,
           at::Tensor v_gate,
           at::Tensor ci_gate,
           at::Tensor v_up,
           at::Tensor ci_up,
           at::Tensor v_down,
           at::Tensor ci_down) -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
            auto result = titan::block_ell::swiglu_backward_cuda(
                grad_out, x, gate_pre_act, up_out, h,
                v_gate, ci_gate,
                v_up,   ci_up,
                v_down, ci_down
            );
            return {
                result.d_x,
                result.d_values_gate,
                result.d_values_up,
                result.d_values_down,
            };
        }
    );
}

// ---------------------------------------------------------------------------
// Meta (FakeTensor) implementations — shape inference for torch.compile.
//
// Dynamo traces with fake tensors that carry shape and dtype but no CUDA memory.
// These functions must return tensors with correct shapes without accessing data.
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(titan, Meta, m) {
    m.impl("block_ell_swiglu_fwd",
        [](at::Tensor x,
           at::Tensor /* v_gate */,
           at::Tensor /* ci_gate */,
           at::Tensor /* v_up */,
           at::Tensor /* ci_up */,
           at::Tensor /* v_down */,
           at::Tensor /* ci_down */,
           bool /* add_residual */) -> at::Tensor {
            // Output has same shape and dtype as x: [N, d_model] bf16
            return at::empty_like(x);
        }
    );

    m.impl("block_ell_swiglu_fwd_with_intermediate",
        [](at::Tensor x,
           at::Tensor v_gate,
           at::Tensor /* ci_gate */,
           at::Tensor /* v_up */,
           at::Tensor /* ci_up */,
           at::Tensor /* v_down */,
           at::Tensor /* ci_down */,
           bool /* add_residual */) -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
            // result: [N, d_model] bf16
            auto result = at::empty_like(x);
            // intermediates: [N, d_ff] bf16 each
            // d_ff = R_ff * TILE = v_gate.size(0) * 16
            const int64_t N    = x.size(0);
            const int64_t d_ff = v_gate.size(0) * 16;
            auto h            = at::empty({N, d_ff}, x.options());
            auto gate_pre_act = at::empty({N, d_ff}, x.options());
            auto up_output    = at::empty({N, d_ff}, x.options());
            return {result, h, gate_pre_act, up_output};
        }
    );

    m.impl("block_ell_swiglu_bwd",
        [](at::Tensor /* grad_out */,
           at::Tensor x,
           at::Tensor /* gate_pre_act */,
           at::Tensor /* up_out */,
           at::Tensor /* h */,
           at::Tensor v_gate,
           at::Tensor /* ci_gate */,
           at::Tensor v_up,
           at::Tensor /* ci_up */,
           at::Tensor v_down,
           at::Tensor /* ci_down */) -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
            // dx: [N, d_model] bf16 — same shape/dtype as x
            auto dx = at::empty_like(x);
            // weight grads: same shape/dtype as the corresponding value tensors
            auto d_vg = at::empty_like(v_gate);
            auto d_vu = at::empty_like(v_up);
            auto d_vd = at::empty_like(v_down);
            return {dx, d_vg, d_vu, d_vd};
        }
    );
}
