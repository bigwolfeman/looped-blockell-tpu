/**
 * memory_ops.cpp — torch.library registration for fused neural memory ops.
 *
 * Registers two custom ops under the `titan::` namespace:
 *   titan::memory_update   — in-place weight update of the memory MLP
 *   titan::memory_retrieve — forward pass through the memory MLP (read-only)
 *
 * Eliminates the torch.autograd.grad() graph break in NeuralMemory.update()
 * by wrapping the entire Titans Eq. 13-14 update as an opaque custom op.
 * Dynamo sees a single call with no Python-level graph break.
 *
 * Memory MLP assumed shape (fixed for CUDA kernel):
 *   4 × Linear layers, each [1024, 1024] weights + [1024] bias
 *   Activations: SiLU between each hidden layer (not passed — baked into kernel)
 *
 * Build: add to CMakeLists.txt / setup.py with CUDA counterpart memory_kernels.cu
 */

#include <torch/library.h>
#include <torch/types.h>
#include <ATen/ATen.h>

// Forward declaration — implemented in memory_kernels.cu
// Full definition lives in memory_kernels.cuh
torch::Tensor memory_update_cuda(
    const torch::Tensor& keys,       // [N, 1024] bf16
    const torch::Tensor& values,     // [N, 1024] bf16
    torch::Tensor& w0,               // [1024, 1024] bf16 — MUTATED
    torch::Tensor& w1,               // [1024, 1024] bf16 — MUTATED
    torch::Tensor& w2,               // [1024, 1024] bf16 — MUTATED
    torch::Tensor& w3,               // [1024, 1024] bf16 — MUTATED
    torch::Tensor& b0,               // [1024] bf16 — MUTATED
    torch::Tensor& b1,               // [1024] bf16 — MUTATED
    torch::Tensor& b2,               // [1024] bf16 — MUTATED
    torch::Tensor& b3,               // [1024] bf16 — MUTATED
    torch::Tensor& momentum,         // [n_params] fp32 — MUTATED
    float eta,                       // momentum decay (typically 0.95)
    float theta,                     // learning rate (typically 0.01)
    float alpha                      // forget rate (0.0001-0.003)
);

torch::Tensor memory_retrieve_cuda(
    const torch::Tensor& query,      // [N, 1024] bf16
    const torch::Tensor& w0,         // [1024, 1024] bf16
    const torch::Tensor& w1,         // [1024, 1024] bf16
    const torch::Tensor& w2,         // [1024, 1024] bf16
    const torch::Tensor& w3,         // [1024, 1024] bf16
    const torch::Tensor& b0,         // [1024] bf16
    const torch::Tensor& b1,         // [1024] bf16
    const torch::Tensor& b2,         // [1024] bf16
    const torch::Tensor& b3          // [1024] bf16
);

// ---------------------------------------------------------------------------
// Schema registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY(titan, m) {
    // memory_update: takes keys, values, mutates 4 weight matrices + 4 biases +
    // momentum buffer in-place, returns loss scalar [1] fp32.
    //
    // Aliasing annotations (a!..i!) tell PyTorch which Tensor args are mutated
    // so that autograd and torch.compile can reason about aliasing correctly.
    m.def(
        "memory_update("
        "    Tensor keys,"
        "    Tensor values,"
        "    Tensor(a!) w0,"
        "    Tensor(b!) w1,"
        "    Tensor(c!) w2,"
        "    Tensor(d!) w3,"
        "    Tensor(e!) b0,"
        "    Tensor(f!) b1,"
        "    Tensor(g!) b2,"
        "    Tensor(h!) b3,"
        "    Tensor(i!) momentum,"
        "    float eta,"
        "    float theta,"
        "    float alpha"
        ") -> Tensor"
    );

    // memory_retrieve: read-only forward pass through the memory MLP.
    // Returns output [N, 1024] bf16.
    m.def(
        "memory_retrieve("
        "    Tensor query,"
        "    Tensor w0,"
        "    Tensor w1,"
        "    Tensor w2,"
        "    Tensor w3,"
        "    Tensor b0,"
        "    Tensor b1,"
        "    Tensor b2,"
        "    Tensor b3"
        ") -> Tensor"
    );
}

// ---------------------------------------------------------------------------
// CUDA implementation dispatch
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(titan, CUDA, m) {
    m.impl("memory_update",
        [](const torch::Tensor& keys,
           const torch::Tensor& values,
           torch::Tensor w0,
           torch::Tensor w1,
           torch::Tensor w2,
           torch::Tensor w3,
           torch::Tensor b0,
           torch::Tensor b1,
           torch::Tensor b2,
           torch::Tensor b3,
           torch::Tensor momentum,
           double eta,
           double theta,
           double alpha) -> torch::Tensor {
            return memory_update_cuda(
                keys, values,
                w0, w1, w2, w3,
                b0, b1, b2, b3,
                momentum,
                static_cast<float>(eta),
                static_cast<float>(theta),
                static_cast<float>(alpha)
            );
        }
    );

    m.impl("memory_retrieve",
        [](const torch::Tensor& query,
           const torch::Tensor& w0,
           const torch::Tensor& w1,
           const torch::Tensor& w2,
           const torch::Tensor& w3,
           const torch::Tensor& b0,
           const torch::Tensor& b1,
           const torch::Tensor& b2,
           const torch::Tensor& b3) -> torch::Tensor {
            return memory_retrieve_cuda(
                query,
                w0, w1, w2, w3,
                b0, b1, b2, b3
            );
        }
    );
}

// ---------------------------------------------------------------------------
// Meta (FakeTensor) implementation — shape inference for torch.compile
//
// Dynamo traces through the model using "fake" tensors that carry only shape
// and dtype (no data). The Meta implementation must return the correct output
// shape/dtype without touching CUDA memory.
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(titan, Meta, m) {
    m.impl("memory_update",
        [](const torch::Tensor& keys,
           const torch::Tensor& /* values */,
           torch::Tensor /* w0 */,
           torch::Tensor /* w1 */,
           torch::Tensor /* w2 */,
           torch::Tensor /* w3 */,
           torch::Tensor /* b0 */,
           torch::Tensor /* b1 */,
           torch::Tensor /* b2 */,
           torch::Tensor /* b3 */,
           torch::Tensor /* momentum */,
           double /* eta */,
           double /* theta */,
           double /* alpha */) -> torch::Tensor {
            // Returns loss scalar [1] fp32 — dtype is fp32 regardless of input dtype
            return torch::empty({1}, keys.options().dtype(torch::kFloat32));
        }
    );

    m.impl("memory_retrieve",
        [](const torch::Tensor& query,
           const torch::Tensor& /* w0 */,
           const torch::Tensor& /* w1 */,
           const torch::Tensor& /* w2 */,
           const torch::Tensor& /* w3 */,
           const torch::Tensor& /* b0 */,
           const torch::Tensor& /* b1 */,
           const torch::Tensor& /* b2 */,
           const torch::Tensor& /* b3 */) -> torch::Tensor {
            // Output is same shape and dtype as query: [N, d_memory] bf16
            return torch::empty_like(query);
        }
    );
}
