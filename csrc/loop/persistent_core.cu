// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// persistent_core.cu — CUDA implementations of injection, RMSNorm, and the
// full persistent MLP step for the Looped Block-ELL core loop.
//
// See persistent_core.cuh for API documentation.
//
// Branch: 006-looped-block-ell

#include "persistent_core.cuh"

#include <cmath>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace titan {
namespace loop {

// ===========================================================================
// Shared utilities
// ===========================================================================

// bf16 softplus:  log(1 + exp(x)).  For large |x| we use the numerically
// stable form: max(x, 0) + log1p(exp(-|x|)).
__device__ __forceinline__ float softplus_f32(float x) {
    // Threshold above which softplus(x) ≈ x (avoids overflow in exp).
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);  // softplus(x) ≈ exp(x) for large neg x
    return log1pf(expf(x));
}

// ===========================================================================
// Kernel: diagonal_injection_kernel
//
// Grid:  ceil(N*d/256) blocks, 256 threads/block.
// Each thread handles one (batch, feature) pair.
//
// Math:
//   dt_i  = softplus(log_dt[i])
//   A_i   = -exp(log_A[i])
//   decay = exp(dt_i * A_i)  = exp(-dt_i * exp(log_A[i]))
//   h[n,i] = decay * h[n,i] + dt_i * e[n,i]
//
// Compute in fp32, store bf16.
// ===========================================================================

__global__ void diagonal_injection_kernel(
    const __nv_bfloat16* __restrict__ h_in,     // [N, d_model] bf16
    const __nv_bfloat16* __restrict__ e,         // [N, d_model] bf16
    const float*          __restrict__ log_A,    // [d_model]    fp32
    const float*          __restrict__ log_dt,   // [d_model]    fp32
    __nv_bfloat16*        __restrict__ h_out,    // [N, d_model] bf16 (output)
    int N,
    int d_model
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * d_model;
    if (idx >= total) return;

    const int feat = idx % d_model;

    // Load scalar params for this feature (same for all N).
    const float lA  = log_A[feat];
    const float ldt = log_dt[feat];
    const float dt  = softplus_f32(ldt);
    // A = -exp(log_A), so dt*A = -dt*exp(log_A)
    const float decay = expf(-dt * expf(lA));

    // Load h and e as fp32 for accumulation.
    const float h_f32 = __bfloat162float(h_in[idx]);
    const float e_f32 = __bfloat162float(e[idx]);

    // h_new = decay * h + dt * e
    const float h_new = decay * h_f32 + dt * e_f32;

    h_out[idx] = __float2bfloat16(h_new);
}

// ===========================================================================
// Kernel: rmsnorm_kernel
//
// One threadblock per row (one token).
// Uses warp-level reduction to compute row RMS.
//
// Grid:  N blocks (one per token).
// Block: min(d_model, 1024) threads.
//
// Accumulation in fp32 to avoid bf16 overflow when squaring activations.
// ===========================================================================

__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,       // [N, d_model] bf16
    const __nv_bfloat16* __restrict__ scale,   // [d_model]    bf16
    __nv_bfloat16*        __restrict__ out,    // [N, d_model] bf16
    int d_model,
    float eps
) {
    // Each block handles one row (one token).
    const int row = blockIdx.x;
    const __nv_bfloat16* x_row   = x   + row * d_model;
          __nv_bfloat16* out_row = out + row * d_model;

    // -----------------------------------------------------------------------
    // Pass 1: compute sum of squares using parallel reduction.
    // -----------------------------------------------------------------------
    float local_sq_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float xi = __bfloat162float(x_row[i]);
        local_sq_sum += xi * xi;
    }

    // Warp-level reduction (shuffle down).
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sq_sum += __shfl_xor_sync(0xffffffff, local_sq_sum, offset);
    }

    // Block-level reduction via shared memory.
    __shared__ float smem[32];  // One slot per warp (max 32 warps per block).
    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    if (lane == 0) smem[wid] = local_sq_sum;
    __syncthreads();

    // Final reduction across warp leaders.
    float sq_sum = 0.0f;
    if (threadIdx.x < (blockDim.x + 31) / 32) {
        sq_sum = smem[threadIdx.x];
    }
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);
        }
    }
    __syncthreads();
    if (wid == 0 && lane == 0) smem[0] = sq_sum;
    __syncthreads();

    const float rms = rsqrtf(smem[0] / static_cast<float>(d_model) + eps);

    // -----------------------------------------------------------------------
    // Pass 2: apply norm and scale.
    // -----------------------------------------------------------------------
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float xi     = __bfloat162float(x_row[i]);
        float si     = __bfloat162float(scale[i]);
        out_row[i]   = __float2bfloat16(xi * rms * si);
    }
}

// ===========================================================================
// Host functions
// ===========================================================================

// ---------------------------------------------------------------------------
// diagonal_injection_cuda
// ---------------------------------------------------------------------------

at::Tensor diagonal_injection_cuda(
    at::Tensor        h,
    const at::Tensor& e,
    const at::Tensor& log_A,
    const at::Tensor& log_dt
) {
    TORCH_CHECK(h.is_cuda(),     "diagonal_injection: h must be on CUDA");
    TORCH_CHECK(e.is_cuda(),     "diagonal_injection: e must be on CUDA");
    TORCH_CHECK(log_A.is_cuda(), "diagonal_injection: log_A must be on CUDA");
    TORCH_CHECK(log_dt.is_cuda(),"diagonal_injection: log_dt must be on CUDA");

    TORCH_CHECK(h.dim() == 2,    "diagonal_injection: h must be 2D [N, d_model]");
    TORCH_CHECK(e.sizes() == h.sizes(),
        "diagonal_injection: e and h must have the same shape");

    const int N       = static_cast<int>(h.size(0));
    const int d_model = static_cast<int>(h.size(1));

    TORCH_CHECK(log_A.numel() == d_model,
        "diagonal_injection: log_A must have d_model elements");
    TORCH_CHECK(log_dt.numel() == d_model,
        "diagonal_injection: log_dt must have d_model elements");
    TORCH_CHECK(h.scalar_type()  == at::kBFloat16, "diagonal_injection: h must be bf16");
    TORCH_CHECK(e.scalar_type()  == at::kBFloat16, "diagonal_injection: e must be bf16");
    TORCH_CHECK(log_A.scalar_type()  == at::kFloat, "diagonal_injection: log_A must be fp32");
    TORCH_CHECK(log_dt.scalar_type() == at::kFloat, "diagonal_injection: log_dt must be fp32");

    // Make contiguous inputs.
    at::Tensor h_c    = h.contiguous();
    at::Tensor e_c    = e.contiguous();
    at::Tensor lA_c   = log_A.contiguous();
    at::Tensor ldt_c  = log_dt.contiguous();

    // Output tensor (same shape/dtype as h).
    at::Tensor h_out = at::empty_like(h_c);

    const int total   = N * d_model;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    diagonal_injection_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(e_c.data_ptr<at::BFloat16>()),
        lA_c.data_ptr<float>(),
        ldt_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(h_out.data_ptr<at::BFloat16>()),
        N,
        d_model
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return h_out;
}

// ---------------------------------------------------------------------------
// rmsnorm_cuda
// ---------------------------------------------------------------------------

at::Tensor rmsnorm_cuda(
    const at::Tensor& x,
    const at::Tensor& scale,
    float             eps
) {
    TORCH_CHECK(x.is_cuda(),     "rmsnorm: x must be on CUDA");
    TORCH_CHECK(scale.is_cuda(), "rmsnorm: scale must be on CUDA");
    TORCH_CHECK(x.dim() == 2,    "rmsnorm: x must be 2D [N, d_model]");
    TORCH_CHECK(x.scalar_type()     == at::kBFloat16, "rmsnorm: x must be bf16");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "rmsnorm: scale must be bf16");

    const int N       = static_cast<int>(x.size(0));
    const int d_model = static_cast<int>(x.size(1));

    TORCH_CHECK(scale.numel() == d_model,
        "rmsnorm: scale must have d_model elements, got ", scale.numel());

    at::Tensor x_c     = x.contiguous();
    at::Tensor scale_c = scale.contiguous();
    at::Tensor out     = at::empty_like(x_c);

    // One block per row; number of threads = min(d_model rounded to next
    // power-of-2, 1024).
    // We use at most 1024 threads per block (SM limit).
    int threads = 256;
    if (d_model > 512)  threads = 512;
    if (d_model > 1024) threads = 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_kernel<<<N, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(scale_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        d_model,
        eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ---------------------------------------------------------------------------
// persistent_mlp_step
//
// Chains: injection (i==0) → RMSNorm → SwiGLU for each core block.
// The Block-ELL SwiGLU forward is delegated to titan::block_ell::swiglu_forward_cuda.
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
) {
    const int n_core = static_cast<int>(norm_scales.size());

    TORCH_CHECK(n_core > 0, "persistent_mlp_step: n_core must be > 0");
    TORCH_CHECK(
        static_cast<int>(gate_values.size())  == n_core &&
        static_cast<int>(gate_col_idx.size()) == n_core &&
        static_cast<int>(up_values.size())    == n_core &&
        static_cast<int>(up_col_idx.size())   == n_core &&
        static_cast<int>(down_values.size())  == n_core &&
        static_cast<int>(down_col_idx.size()) == n_core,
        "persistent_mlp_step: all weight vector sizes must equal n_core"
    );

    TORCH_CHECK(h.dim() == 2,
        "persistent_mlp_step: h must be 2D [N, d_model], got shape ", h.sizes());
    TORCH_CHECK(h.scalar_type() == at::kBFloat16,
        "persistent_mlp_step: h must be bf16");

    // -----------------------------------------------------------------------
    // Step 1: Diagonal SSM injection (only once, at the start of the MLP pass).
    // This fuses the hidden state from the previous SDPA output with the
    // injection signal e.
    // -----------------------------------------------------------------------
    at::Tensor h_cur = diagonal_injection_cuda(h, e, log_A, log_dt);

    // -----------------------------------------------------------------------
    // Step 2: Core blocks — RMSNorm + Block-ELL SwiGLU.
    // Pre-Norm convention: h_cur = h_cur + MLP(RMSNorm(h_cur))
    // -----------------------------------------------------------------------
    for (int i = 0; i < n_core; ++i) {
        // Pre-MLP RMSNorm — normalised input fed to MLP, original h_cur is the skip.
        at::Tensor h_normed = rmsnorm_cuda(h_cur, norm_scales[i], 1e-6f);

        // Block-ELL SwiGLU MLP — no fused residual here because the residual
        // base is h_cur (pre-norm), not h_normed (post-norm).
        // add_residual=false: we add the skip connection manually below.
        at::Tensor mlp_out = titan::block_ell::swiglu_forward_cuda(
            h_normed,
            gate_values[i],
            gate_col_idx[i],
            up_values[i],
            up_col_idx[i],
            down_values[i],
            down_col_idx[i],
            /*add_residual=*/false
        );

        // Pre-Norm skip connection: h_cur = MLP(norm(h_cur)) + h_cur.
        h_cur = mlp_out.add(h_cur);
    }

    return h_cur;
}

} // namespace loop
} // namespace titan
