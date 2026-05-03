/**
 * loop_ops.cpp — torch.library registration for L2-pinned persistent loop ops.
 *
 * Registers three custom ops under the `titan::` namespace:
 *
 *   titan::pin_weights_l2(Tensor[] weights) -> ()
 *     Pin weight tensors in L2 persistent cache on the current CUDA stream.
 *     Call once before the training loop.  Re-call after topology compaction.
 *
 *   titan::unpin_weights_l2() -> ()
 *     Release the L2 persistent cache reservation.
 *
 *   titan::persistent_mlp_step(h, e, log_A, log_dt,
 *                               norm_scales, gate_values, gate_col_idx,
 *                               up_values, up_col_idx,
 *                               down_values, down_col_idx) -> Tensor
 *     One iteration of the MLP path of the core loop:
 *       injection → (RMSNorm + Block-ELL SwiGLU) × n_core
 *
 * Build: add loop_ops.cpp, l2_policy.cu, persistent_core.cu to setup.py sources.
 */

#include <torch/library.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declarations from l2_policy.cu and persistent_core.cu
#include "l2_policy.cuh"
#include "persistent_core.cuh"

// ---------------------------------------------------------------------------
// Schema registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(titan, m) {
    // pin_weights_l2: takes a list of weight tensors, returns nothing.
    // The stream is taken implicitly from the current CUDA stream context.
    m.def(
        "pin_weights_l2("
        "    Tensor[] weights"
        ") -> ()"
    );

    // unpin_weights_l2: takes nothing, returns nothing.
    m.def("unpin_weights_l2() -> ()");

    // persistent_mlp_step: injection + (RMSNorm + SwiGLU) × n_core.
    //
    // The per-block weight lists (norm_scales, *_values, *_col_idx) each have
    // n_core elements.  torch.library TensorList maps to std::vector<Tensor>.
    m.def(
        "persistent_mlp_step("
        "    Tensor h,"               // [N, d_model] bf16 current hidden state
        "    Tensor e,"               // [N, d_model] bf16 injection input
        "    Tensor log_A,"           // [d_model]    fp32 SSM A (log scale)
        "    Tensor log_dt,"          // [d_model]    fp32 SSM dt (log scale)
        "    Tensor[] norm_scales,"   // n_core × [d_model]       bf16
        "    Tensor[] gate_values,"   // n_core × [R_ff, K, 16, 16] bf16
        "    Tensor[] gate_col_idx,"  // n_core × [R_ff, K]         int32
        "    Tensor[] up_values,"     // n_core × [R_ff, K, 16, 16] bf16
        "    Tensor[] up_col_idx,"    // n_core × [R_ff, K]         int32
        "    Tensor[] down_values,"   // n_core × [R_model, K, 16, 16] bf16
        "    Tensor[] down_col_idx"   // n_core × [R_model, K]         int32
        ") -> Tensor"
    );
}

// ---------------------------------------------------------------------------
// CUDA implementation dispatch
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(titan, CUDA, m) {
    m.impl("pin_weights_l2",
        [](const std::vector<at::Tensor>& weights) -> void {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            titan::loop::pin_weights_l2(weights, stream);
        }
    );

    m.impl("unpin_weights_l2",
        []() -> void {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            titan::loop::unpin_weights_l2(stream);
        }
    );

    m.impl("persistent_mlp_step",
        [](at::Tensor                    h,
           at::Tensor                    e,
           at::Tensor                    log_A,
           at::Tensor                    log_dt,
           std::vector<at::Tensor>       norm_scales,
           std::vector<at::Tensor>       gate_values,
           std::vector<at::Tensor>       gate_col_idx,
           std::vector<at::Tensor>       up_values,
           std::vector<at::Tensor>       up_col_idx,
           std::vector<at::Tensor>       down_values,
           std::vector<at::Tensor>       down_col_idx
        ) -> at::Tensor {
            return titan::loop::persistent_mlp_step(
                std::move(h),
                e,
                log_A,
                log_dt,
                std::move(norm_scales),
                std::move(gate_values),
                std::move(gate_col_idx),
                std::move(up_values),
                std::move(up_col_idx),
                std::move(down_values),
                std::move(down_col_idx)
            );
        }
    );
}

// ---------------------------------------------------------------------------
// Meta (FakeTensor) implementations — shape inference for torch.compile.
//
// Dynamo traces with fake tensors that carry shape/dtype but no data.
// The Meta kernel must return the correct output shape without touching CUDA.
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(titan, Meta, m) {
    // pin/unpin are side effects — no tensor output.
    m.impl("pin_weights_l2",
        [](const std::vector<at::Tensor>& /* weights */) -> void {
            // No-op for fake tensor tracing.
        }
    );

    m.impl("unpin_weights_l2",
        []() -> void {
            // No-op for fake tensor tracing.
        }
    );

    m.impl("persistent_mlp_step",
        [](at::Tensor                    h,
           at::Tensor                    /* e */,
           at::Tensor                    /* log_A */,
           at::Tensor                    /* log_dt */,
           std::vector<at::Tensor>       /* norm_scales */,
           std::vector<at::Tensor>       /* gate_values */,
           std::vector<at::Tensor>       /* gate_col_idx */,
           std::vector<at::Tensor>       /* up_values */,
           std::vector<at::Tensor>       /* up_col_idx */,
           std::vector<at::Tensor>       /* down_values */,
           std::vector<at::Tensor>       /* down_col_idx */
        ) -> at::Tensor {
            // Output has same shape and dtype as h.
            return at::empty_like(h);
        }
    );
}
