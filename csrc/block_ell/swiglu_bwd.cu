// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// swiglu_bwd.cu — Fused Block-ELL SwiGLU backward pass implementation.
//
// See swiglu_bwd.cuh for the full parameter contract and math description.
//
// Five-kernel design (two sub-kernels each for down and gate/up):
//
//   Kernel 1a  block_ell_down_backward_dh
//              Scatters d_h = grad_output @ values_down^T (per tile, atomicAdd).
//              Grid (R_model, ceil(N/BLOCK_N)).  One warp per 16 tokens × WMMA tile.
//
//   Kernel 1b  block_ell_down_backward_dw
//              d_W_down[r,k] = h[:,col]^T @ grad_output[:,r]  (fp32 outer product).
//              Grid (R_model * K_d,).  One warp per tile, loops over token batches.
//
//   Kernel 2   silu_mul_backward_kernel
//              Elementwise backward through h = silu(gate) * up.
//              Computes d_gate and d_up.  Grid (ceil(N*d_ff/256),).
//
//   Kernel 3a  block_ell_proj_backward_dx
//              Scatters d_x from d_gate and d_up via gate/up weight transposes.
//              Grid (R_ff*(K_g+K_u), ceil(N/BLOCK_N)).  atomicAdd to d_x.
//
//   Kernel 3b  residual_add_kernel
//              d_x += grad_output  (backward through residual add).
//              Grid (ceil(N*d_model/256),).
//
//   Kernel 3c  block_ell_proj_backward_dw
//              d_W_gate[r,k] = x[:,c]^T @ d_gate[:,r]  and same for up.
//              Grid (R_ff*(K_g+K_u),).  One warp per tile, loops over token batches.
//
// All weight gradients are accumulated in fp32 and cast to bf16 before return.
// Input/hidden gradients use bf16 atomicAdd (native on sm_90a+).
// WMMA fragment pairs: __nv_bfloat16 A/B, float accumulator.
//
// Branch: 006-looped-block-ell

#include "swiglu_bwd.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <mma.h>
#include <cuda_bf16.h>

using namespace nvcuda;

namespace titan {
namespace block_ell {

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

// Block-ELL tile size — must match the format (always 16 in this project)
static constexpr int TILE = 16;

// Token batch size per CTA for scatter kernels.
// Must be a multiple of TILE=16 for WMMA.
// 32 keeps shared memory under 8KB per CTA across all kernels here.
static constexpr int BLOCK_N = 32;

// Threads per block for flat elementwise kernels (Kernels 2 and 3b)
static constexpr int BLOCK_EW = 256;

// Number of warps tiling the BLOCK_N token dimension in scatter kernels
static constexpr int WARPS_N = BLOCK_N / TILE;   // = 2

// Threads per block for scatter kernels (two warps × 32 lanes)
static constexpr int THREADS_SCATTER = WARPS_N * 32;  // = 64

// Threads per block for dW kernels (single warp — one WMMA per tile is enough)
static constexpr int THREADS_DW = 32;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// silu'(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
// Pass pre-computed sig to avoid recomputing the exponential.
__device__ __forceinline__ float silu_prime(float x, float sig) {
    return sig + x * sig * (1.0f - sig);
}

// Add a float value into a bf16 address using atomicAdd.
// Native bf16 atomicAdd is available on sm_90a+; falls back to CAS loop on
// older hardware (not needed for our target, but included for correctness).
__device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16* addr, float val) {
    atomicAdd(addr, __float2bfloat16(val));
}

// ---------------------------------------------------------------------------
// Kernel 1a: block_ell_down_backward_dh
//
// Computes:
//   d_h[:, c*TILE:(c+1)*TILE] += grad_output[:, r*TILE:(r+1)*TILE] @ values_down[r,k]
//
// for all (r, k) in the down projection, where c = col_idx_down[r, k].
//
// The down projection forward was:  out = h @ W_down^T
// So its backward w.r.t. h is:      d_h = grad_out @ W_down     (no transpose)
// In block form for one tile:
//   d_h_block[col]  += grad_out_block[r] @ values_down[r,k]
//   [BLOCK_N × TILE]  = [BLOCK_N × TILE] @ [TILE × TILE]
//
// Multiple (r,k) pairs can map to the same col → scatter-add with atomicAdd.
//
// Grid: (R_model, ceil(N / BLOCK_N))
//   blockIdx.x = r   (output block-row index of the down projection)
//   blockIdx.y = token batch index
//
// blockDim.x = THREADS_SCATTER = WARPS_N * 32
//   warp 0 handles tokens [0,   TILE)
//   warp 1 handles tokens [TILE, BLOCK_N)
//
// Shared memory layout:
//   smem_weight: [TILE * TILE]   bf16  — weight tile values_down[r,k]
//   smem_go:     [BLOCK_N * TILE] bf16  — grad_output rows for this CTA
//   smem_acc:    [BLOCK_N * TILE] float — WMMA accumulator staging area
// ---------------------------------------------------------------------------
__global__ void block_ell_down_backward_dh(
    const __nv_bfloat16* __restrict__ grad_out,   // [N, d_model]
    int64_t stride_go_n,
    const __nv_bfloat16* __restrict__ values_down, // [R_model, K_d, TILE, TILE]
    int64_t stride_vd_r,
    int64_t stride_vd_k,
    const int*           __restrict__ col_idx_down, // [R_model, K_d]
    int64_t stride_ci_r,
    __nv_bfloat16*       __restrict__ d_h,          // [N, d_ff]  zero-init
    int64_t stride_dh_n,
    int N,
    int K_d)
{
    const int r     = blockIdx.x;
    const int pid_n = blockIdx.y;
    const int warp  = threadIdx.x / 32;
    const int lane  = threadIdx.x % 32;

    // Shared memory regions (non-overlapping)
    __shared__ __nv_bfloat16 smem_weight[TILE * TILE];
    __shared__ __nv_bfloat16 smem_go[BLOCK_N * TILE];
    __shared__ float          smem_acc[BLOCK_N * TILE];

    // Load grad_output block: rows [pid_n*BLOCK_N : (pid_n+1)*BLOCK_N],
    //                          cols [r*TILE : (r+1)*TILE]
    // Stride: BLOCK_N * TILE elements, cooperative load across all threads.
    for (int t = threadIdx.x; t < BLOCK_N * TILE; t += blockDim.x) {
        int tok = pid_n * BLOCK_N + t / TILE;
        int col = r * TILE + (t % TILE);
        smem_go[t] = (tok < N) ? grad_out[tok * stride_go_n + col]
                               : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Process each K_d tile independently.
    // The WMMA accumulator is reset each iteration and scatter-added after.
    for (int k = 0; k < K_d; ++k) {
        const int c = col_idx_down[r * stride_ci_r + k];

        // Load weight tile: values_down[r, k, :, :] — 256 elements, load with first 256 threads
        if (threadIdx.x < TILE * TILE) {
            smem_weight[threadIdx.x] =
                values_down[r * stride_vd_r + k * stride_vd_k + threadIdx.x];
        }
        __syncthreads();

        // Each warp computes one [TILE × TILE] WMMA block for its token slice.
        if (warp < WARPS_N) {
            wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                           __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            // a_frag: grad_out rows for this warp's 16 tokens — row_major [TILE, TILE]
            wmma::load_matrix_sync(a_frag, smem_go + warp * (TILE * TILE), TILE);

            // b_frag: weight loaded col_major = implicit transpose
            // values_down[r,k] is [TILE_out, TILE_in]; col_major gives ^T = [TILE_in, TILE_out]
            // so a_frag @ b_frag^T is correct: [TILE, TILE_out] @ [TILE_out, TILE_in]^T
            // = [TILE, TILE_in] = d_h contribution
            wmma::load_matrix_sync(b_frag, smem_weight, TILE);

            wmma::mma_sync(acc, a_frag, b_frag, acc);

            // Stage result to shared memory for scatter
            wmma::store_matrix_sync(smem_acc + warp * (TILE * TILE), acc, TILE,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        // Scatter-add from smem_acc into d_h at column block c
        if (warp < WARPS_N) {
            for (int t = lane; t < TILE * TILE; t += 32) {
                const int tok_local  = t / TILE;
                const int feat_local = t % TILE;
                const int tok_global = pid_n * BLOCK_N + warp * TILE + tok_local;
                if (tok_global < N) {
                    atomic_add_bf16(
                        d_h + tok_global * stride_dh_n + c * TILE + feat_local,
                        smem_acc[warp * (TILE * TILE) + t]);
                }
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel 1b: block_ell_down_backward_dw
//
// Computes:
//   d_W_down[r, k] = h[:, c*TILE:(c+1)*TILE]^T @ grad_output[:, r*TILE:(r+1)*TILE]
//   where c = col_idx_down[r, k]
//
// Forward was: out = h @ W_down^T, so d_W_down = grad_out^T @ h
// In block form: d_W_down[r,k] = grad_out_block[r]^T @ h_block[col]
//                [TILE × TILE]  = [TILE × N] @ [N × TILE]  (sum over N tokens)
//
// WMMA strategy:
//   matrix_a (col_major) ← grad_out_block^T      : [TILE, BLOCK_N] in col_major layout
//   matrix_b (row_major) ← h_block                : [BLOCK_N, TILE] in row_major layout
//   accumulator           ← d_W_down[r,k]         : [TILE, TILE] fp32
//
// Grid: (R_model * K_d,)  — one CTA per (r,k) tile
//   Inner loop over N in BLOCK_N chunks.
//
// blockDim.x = THREADS_DW = 32  (one warp)
// ---------------------------------------------------------------------------
__global__ void block_ell_down_backward_dw(
    const __nv_bfloat16* __restrict__ h,          // [N, d_ff]  intermediate hidden
    int64_t stride_h_n,
    const __nv_bfloat16* __restrict__ grad_out,   // [N, d_model]
    int64_t stride_go_n,
    const int*           __restrict__ col_idx_down, // [R_model, K_d]
    int64_t stride_ci_r,
    float*               __restrict__ d_values_down, // [R_model, K_d, TILE, TILE] fp32
    int64_t stride_dv_r,
    int64_t stride_dv_k,
    int N,
    int K_d)
{
    const int rk = blockIdx.x;
    const int r  = rk / K_d;
    const int k  = rk % K_d;
    const int c  = col_idx_down[r * stride_ci_r + k];

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                   __nv_bfloat16, wmma::col_major> a_frag;  // grad_out^T
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                   __nv_bfloat16, wmma::row_major> b_frag;  // h block
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    __shared__ __nv_bfloat16 smem_go[BLOCK_N * TILE];  // grad_out chunk
    __shared__ __nv_bfloat16 smem_h [BLOCK_N * TILE];  // h chunk

    for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
        // Load grad_out chunk: [n0:n0+BLOCK_N, r*TILE:(r+1)*TILE]
        for (int t = threadIdx.x; t < BLOCK_N * TILE; t += blockDim.x) {
            const int tok = n0 + t / TILE;
            const int col = r * TILE + (t % TILE);
            smem_go[t] = (tok < N) ? grad_out[tok * stride_go_n + col]
                                   : __float2bfloat16(0.0f);
        }
        // Load h chunk: [n0:n0+BLOCK_N, c*TILE:(c+1)*TILE]
        for (int t = threadIdx.x; t < BLOCK_N * TILE; t += blockDim.x) {
            const int tok = n0 + t / TILE;
            const int col = c * TILE + (t % TILE);
            smem_h[t] = (tok < N) ? h[tok * stride_h_n + col]
                                  : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // One warp handles the 16×16 tile.
        // a_frag loaded col_major from smem_go: WMMA interprets smem_go
        // (which is [BLOCK_N, TILE] row-major) as [TILE, BLOCK_N] when col_major
        // is specified — this gives us the implicit transpose we want.
        wmma::load_matrix_sync(a_frag, smem_go, BLOCK_N);
        wmma::load_matrix_sync(b_frag, smem_h,  TILE);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    // Store fp32 accumulator into d_values_down
    __shared__ float smem_acc[TILE * TILE];
    wmma::store_matrix_sync(smem_acc, acc, TILE, wmma::mem_row_major);
    __syncwarp();
    for (int t = threadIdx.x; t < TILE * TILE; t += blockDim.x) {
        d_values_down[r * stride_dv_r + k * stride_dv_k + t] = smem_acc[t];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: silu_mul_backward_kernel
//
// Backward through h = silu(gate_pre_act) * up_output.
//
// Given d_h (gradient w.r.t. h):
//   sig    = sigmoid(gate_pre_act)
//   d_gate = d_h * up_output * (sig + gate_pre_act * sig * (1 - sig))
//   d_up   = d_h * gate_pre_act * sig      [= d_h * silu(gate)]
//
// Pure elementwise — one thread per element.
// Grid: (ceil(N * d_ff / BLOCK_EW),)
// ---------------------------------------------------------------------------
__global__ void silu_mul_backward_kernel(
    const __nv_bfloat16* __restrict__ d_h,           // [N, d_ff]
    const __nv_bfloat16* __restrict__ gate_pre_act,  // [N, d_ff]
    const __nv_bfloat16* __restrict__ up_output,     // [N, d_ff]
    __nv_bfloat16*       __restrict__ d_gate,        // [N, d_ff]  output
    __nv_bfloat16*       __restrict__ d_up,          // [N, d_ff]  output
    int total_elems)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    const float dh   = __bfloat162float(d_h[idx]);
    const float gate = __bfloat162float(gate_pre_act[idx]);
    const float up   = __bfloat162float(up_output[idx]);

    const float sig  = sigmoid_f(gate);          // sigmoid(gate)
    const float sg   = gate * sig;               // silu(gate)

    // d_gate: chain rule through silu, then through elementwise multiply with up
    d_gate[idx] = __float2bfloat16(dh * up * silu_prime(gate, sig));

    // d_up: chain rule through elementwise multiply with silu(gate)
    d_up[idx]   = __float2bfloat16(dh * sg);
}

// ---------------------------------------------------------------------------
// Kernel 3a: block_ell_proj_backward_dx
//
// Computes d_x contributions from d_gate (gate projection backward) and
// d_up (up projection backward) in a single kernel launch.
//
// For each (r, k) pair the kernel handles:
//   if is_gate:
//     c = col_idx_gate[r, k]
//     d_x[:, c*TILE:(c+1)*TILE] += d_gate[:, r*TILE:(r+1)*TILE] @ values_gate[r,k]
//   else:
//     c = col_idx_up[r, k]
//     d_x[:, c*TILE:(c+1)*TILE] += d_up[:, r*TILE:(r+1)*TILE] @ values_up[r,k]
//
// The gate and up halves are folded into one grid:
//   rk in [0,         R_ff*K_g)  → gate projection
//   rk in [R_ff*K_g,  R_ff*(K_g+K_u))  → up projection
//
// Grid: (R_ff*(K_g+K_u), ceil(N/BLOCK_N))
// blockDim.x = THREADS_SCATTER = WARPS_N * 32
// ---------------------------------------------------------------------------
__global__ void block_ell_proj_backward_dx(
    const __nv_bfloat16* __restrict__ d_gate,       // [N, d_ff]
    int64_t stride_dg_n,
    const __nv_bfloat16* __restrict__ d_up,         // [N, d_ff]
    int64_t stride_du_n,
    const __nv_bfloat16* __restrict__ values_gate,  // [R_ff, K_g, TILE, TILE]
    int64_t stride_vg_r,
    int64_t stride_vg_k,
    const int*           __restrict__ col_idx_gate,  // [R_ff, K_g]
    int64_t stride_cig_r,
    const __nv_bfloat16* __restrict__ values_up,    // [R_ff, K_u, TILE, TILE]
    int64_t stride_vu_r,
    int64_t stride_vu_k,
    const int*           __restrict__ col_idx_up,    // [R_ff, K_u]
    int64_t stride_ciu_r,
    __nv_bfloat16*       __restrict__ d_x,           // [N, d_model]  zero-init, scatter target
    int64_t stride_dx_n,
    int N,
    int R_ff,
    int K_g,
    int K_u)
{
    const int rk    = blockIdx.x;
    const int pid_n = blockIdx.y;
    const int warp  = threadIdx.x / 32;
    const int lane  = threadIdx.x % 32;

    // Determine whether this CTA handles the gate or up half
    const bool is_gate = (rk < R_ff * K_g);
    const int  local_rk = is_gate ? rk : (rk - R_ff * K_g);
    const int  K        = is_gate ? K_g : K_u;
    const int  r        = local_rk / K;
    const int  k        = local_rk % K;

    // Select the right arrays
    const __nv_bfloat16* d_proj;
    int64_t stride_dp_n;
    const __nv_bfloat16* values;
    int64_t stride_v_r, stride_v_k;
    int c;

    if (is_gate) {
        d_proj      = d_gate;
        stride_dp_n = stride_dg_n;
        values      = values_gate;
        stride_v_r  = stride_vg_r;
        stride_v_k  = stride_vg_k;
        c           = col_idx_gate[r * stride_cig_r + k];
    } else {
        d_proj      = d_up;
        stride_dp_n = stride_du_n;
        values      = values_up;
        stride_v_r  = stride_vu_r;
        stride_v_k  = stride_vu_k;
        c           = col_idx_up[r * stride_ciu_r + k];
    }

    __shared__ __nv_bfloat16 smem_weight[TILE * TILE];
    __shared__ __nv_bfloat16 smem_dp[BLOCK_N * TILE];
    __shared__ float          smem_acc[BLOCK_N * TILE];

    // Load weight tile
    if (threadIdx.x < TILE * TILE) {
        smem_weight[threadIdx.x] =
            values[r * stride_v_r + k * stride_v_k + threadIdx.x];
    }
    // Load d_proj block: [pid_n*BLOCK_N : (pid_n+1)*BLOCK_N, r*TILE : (r+1)*TILE]
    for (int t = threadIdx.x; t < BLOCK_N * TILE; t += blockDim.x) {
        const int tok = pid_n * BLOCK_N + t / TILE;
        const int col = r * TILE + (t % TILE);
        smem_dp[t] = (tok < N) ? d_proj[tok * stride_dp_n + col]
                               : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Each warp: d_proj_block[warp] @ weight^T → d_x contribution
    if (warp < WARPS_N) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                       __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        wmma::load_matrix_sync(a_frag, smem_dp + warp * (TILE * TILE), TILE);
        wmma::load_matrix_sync(b_frag, smem_weight, TILE);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        wmma::store_matrix_sync(smem_acc + warp * (TILE * TILE), acc, TILE,
                                wmma::mem_row_major);
    }
    __syncthreads();

    // Scatter-add result into d_x
    if (warp < WARPS_N) {
        for (int t = lane; t < TILE * TILE; t += 32) {
            const int tok_local  = t / TILE;
            const int feat_local = t % TILE;
            const int tok_global = pid_n * BLOCK_N + warp * TILE + tok_local;
            if (tok_global < N) {
                atomic_add_bf16(
                    d_x + tok_global * stride_dx_n + c * TILE + feat_local,
                    smem_acc[warp * (TILE * TILE) + t]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3b: residual_add_kernel
//
// Folds in the residual gradient: d_x += grad_output.
// (Backward of result = out + x through the identity shortcut.)
//
// Must run after block_ell_proj_backward_dx has finished writing d_x.
// Grid: (ceil(N * d_model / BLOCK_EW),)
// ---------------------------------------------------------------------------
__global__ void residual_add_kernel(
    const __nv_bfloat16* __restrict__ grad_out,  // [N, d_model]
    __nv_bfloat16*       __restrict__ d_x,        // [N, d_model]  in-place add
    int total_elems)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    d_x[idx] = __float2bfloat16(
        __bfloat162float(d_x[idx]) + __bfloat162float(grad_out[idx]));
}

// ---------------------------------------------------------------------------
// Kernel 3c: block_ell_proj_backward_dw
//
// Computes d_W_gate and d_W_up (weight gradients for the gate and up projections).
//
// For gate:
//   d_W_gate[r, k] = x[:, c*TILE:(c+1)*TILE]^T @ d_gate[:, r*TILE:(r+1)*TILE]
// For up:
//   d_W_up[r, k]   = x[:, c*TILE:(c+1)*TILE]^T @ d_up[:, r*TILE:(r+1)*TILE]
//
// Forward was y = x @ W^T, so d_W = d_out^T @ x
// In block form: d_W[r,k] = d_proj_block[r]^T @ x_block[col]
//                [TILE × TILE] = [TILE × N] @ [N × TILE]  (sum over tokens)
//
// Grid: (R_ff*(K_g+K_u),)  — same gate/up split as Kernel 3a
// blockDim.x = THREADS_DW = 32 (one warp)
// ---------------------------------------------------------------------------
__global__ void block_ell_proj_backward_dw(
    const __nv_bfloat16* __restrict__ x,           // [N, d_model]
    int64_t stride_x_n,
    const __nv_bfloat16* __restrict__ d_gate,      // [N, d_ff]
    int64_t stride_dg_n,
    const __nv_bfloat16* __restrict__ d_up,        // [N, d_ff]
    int64_t stride_du_n,
    const int*           __restrict__ col_idx_gate,  // [R_ff, K_g]
    int64_t stride_cig_r,
    const int*           __restrict__ col_idx_up,    // [R_ff, K_u]
    int64_t stride_ciu_r,
    float*               __restrict__ d_values_gate, // [R_ff, K_g, TILE, TILE] fp32
    int64_t stride_dvg_r,
    int64_t stride_dvg_k,
    float*               __restrict__ d_values_up,   // [R_ff, K_u, TILE, TILE] fp32
    int64_t stride_dvu_r,
    int64_t stride_dvu_k,
    int N,
    int R_ff,
    int K_g,
    int K_u)
{
    const int rk      = blockIdx.x;
    const bool is_gate = (rk < R_ff * K_g);
    const int  local_rk = is_gate ? rk : (rk - R_ff * K_g);
    const int  K        = is_gate ? K_g : K_u;
    const int  r        = local_rk / K;
    const int  k        = local_rk % K;

    const __nv_bfloat16* d_proj;
    int64_t stride_dp_n;
    float* d_values;
    int64_t stride_dv_r, stride_dv_k;
    int c;

    if (is_gate) {
        d_proj      = d_gate;
        stride_dp_n = stride_dg_n;
        d_values    = d_values_gate;
        stride_dv_r = stride_dvg_r;
        stride_dv_k = stride_dvg_k;
        c           = col_idx_gate[r * stride_cig_r + k];
    } else {
        d_proj      = d_up;
        stride_dp_n = stride_du_n;
        d_values    = d_values_up;
        stride_dv_r = stride_dvu_r;
        stride_dv_k = stride_dvu_k;
        c           = col_idx_up[r * stride_ciu_r + k];
    }

    // WMMA: d_W = d_proj_block^T @ x_block
    //   a_frag (col_major) ← d_proj_block [BLOCK_N, TILE] stored as col_major → ^T = [TILE, BLOCK_N]
    //   b_frag (row_major) ← x_block      [BLOCK_N, TILE]
    //   acc                ← [TILE, TILE] fp32
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                   __nv_bfloat16, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                   __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    __shared__ __nv_bfloat16 smem_dp[BLOCK_N * TILE];
    __shared__ __nv_bfloat16 smem_x [BLOCK_N * TILE];

    for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
        // Load d_proj chunk: [n0:n0+BLOCK_N, r*TILE:(r+1)*TILE]
        for (int t = threadIdx.x; t < BLOCK_N * TILE; t += blockDim.x) {
            const int tok = n0 + t / TILE;
            const int col = r * TILE + (t % TILE);
            smem_dp[t] = (tok < N) ? d_proj[tok * stride_dp_n + col]
                                   : __float2bfloat16(0.0f);
        }
        // Load x chunk: [n0:n0+BLOCK_N, c*TILE:(c+1)*TILE]
        for (int t = threadIdx.x; t < BLOCK_N * TILE; t += blockDim.x) {
            const int tok = n0 + t / TILE;
            const int col = c * TILE + (t % TILE);
            smem_x[t] = (tok < N) ? x[tok * stride_x_n + col]
                                  : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // a_frag col_major from smem_dp → implicit d_proj^T
        wmma::load_matrix_sync(a_frag, smem_dp, BLOCK_N);
        wmma::load_matrix_sync(b_frag, smem_x,  TILE);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    __shared__ float smem_acc[TILE * TILE];
    wmma::store_matrix_sync(smem_acc, acc, TILE, wmma::mem_row_major);
    __syncwarp();
    for (int t = threadIdx.x; t < TILE * TILE; t += blockDim.x) {
        d_values[r * stride_dv_r + k * stride_dv_k + t] = smem_acc[t];
    }
}

// ---------------------------------------------------------------------------
// Host function
// ---------------------------------------------------------------------------

SwiGLUBackwardResult swiglu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor gate_pre_act,
    torch::Tensor up_output,
    torch::Tensor h,
    torch::Tensor values_gate,
    torch::Tensor col_idx_gate,
    torch::Tensor values_up,
    torch::Tensor col_idx_up,
    torch::Tensor values_down,
    torch::Tensor col_idx_down)
{
    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be on CUDA");
    TORCH_CHECK(x.is_cuda(),           "x must be on CUDA");
    TORCH_CHECK(gate_pre_act.is_cuda(), "gate_pre_act must be on CUDA");
    TORCH_CHECK(up_output.is_cuda(),    "up_output must be on CUDA");
    TORCH_CHECK(h.is_cuda(),            "h must be on CUDA");

    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D [N, d_model]");
    TORCH_CHECK(x.dim() == 2,           "x must be 2D [N, d_model]");
    TORCH_CHECK(gate_pre_act.dim() == 2,"gate_pre_act must be 2D [N, d_ff]");
    TORCH_CHECK(up_output.dim() == 2,   "up_output must be 2D [N, d_ff]");
    TORCH_CHECK(h.dim() == 2,           "h must be 2D [N, d_ff]");

    TORCH_CHECK(values_down.dim() == 4, "values_down must be 4D [R_model, K_d, 16, 16]");
    TORCH_CHECK(values_gate.dim() == 4, "values_gate must be 4D [R_ff, K_g, 16, 16]");
    TORCH_CHECK(values_up.dim() == 4,   "values_up must be 4D [R_ff, K_u, 16, 16]");

    // -----------------------------------------------------------------------
    // Dimensions
    // -----------------------------------------------------------------------
    const int N       = static_cast<int>(x.size(0));
    const int d_model = static_cast<int>(x.size(1));
    const int d_ff    = static_cast<int>(gate_pre_act.size(1));
    const int R_model = static_cast<int>(values_down.size(0));
    const int K_d     = static_cast<int>(values_down.size(1));
    const int R_ff    = static_cast<int>(values_gate.size(0));
    const int K_g     = static_cast<int>(values_gate.size(1));
    const int K_u     = static_cast<int>(values_up.size(1));

    TORCH_CHECK(N % TILE == 0,
        "N must be divisible by TILE=16 for WMMA (got N=", N, ")");
    TORCH_CHECK(values_down.size(2) == TILE && values_down.size(3) == TILE,
        "values_down tile must be 16×16");
    TORCH_CHECK(values_gate.size(2) == TILE && values_gate.size(3) == TILE,
        "values_gate tile must be 16×16");
    TORCH_CHECK(values_up.size(2) == TILE && values_up.size(3) == TILE,
        "values_up tile must be 16×16");

    // -----------------------------------------------------------------------
    // Ensure contiguous
    // -----------------------------------------------------------------------
    grad_output  = grad_output.contiguous();
    x            = x.contiguous();
    gate_pre_act = gate_pre_act.contiguous();
    up_output    = up_output.contiguous();
    h            = h.contiguous();
    values_gate  = values_gate.contiguous();
    col_idx_gate = col_idx_gate.contiguous();
    values_up    = values_up.contiguous();
    col_idx_up   = col_idx_up.contiguous();
    values_down  = values_down.contiguous();
    col_idx_down = col_idx_down.contiguous();

    // -----------------------------------------------------------------------
    // Allocate intermediate and output tensors
    // -----------------------------------------------------------------------
    const auto opts_bf16 = x.options();
    const auto opts_fp32 = x.options().dtype(torch::kFloat32);

    // d_h: gradient w.r.t. intermediate hidden state h
    // Zero-init because it is populated via atomicAdd scatter from Kernel 1a
    torch::Tensor d_h = torch::zeros({N, d_ff}, opts_bf16);

    // fp32 weight gradient buffers (cast to bf16 on return)
    torch::Tensor d_values_down_f32 = torch::zeros({R_model, K_d, TILE, TILE}, opts_fp32);
    torch::Tensor d_values_gate_f32 = torch::zeros({R_ff,    K_g,  TILE, TILE}, opts_fp32);
    torch::Tensor d_values_up_f32   = torch::zeros({R_ff,    K_u,  TILE, TILE}, opts_fp32);

    // d_gate, d_up: gradients w.r.t. gate and up projections
    torch::Tensor d_gate = torch::empty({N, d_ff}, opts_bf16);
    torch::Tensor d_up   = torch::empty({N, d_ff}, opts_bf16);

    // d_x: zero-init because Kernel 3a populates via atomicAdd
    torch::Tensor d_x = torch::zeros({N, d_model}, opts_bf16);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // -----------------------------------------------------------------------
    // Kernel 1a: down projection backward → d_h
    // Grid (R_model, ceil(N/BLOCK_N)), Block THREADS_SCATTER
    // -----------------------------------------------------------------------
    {
        const dim3 grid(R_model, (N + BLOCK_N - 1) / BLOCK_N);
        block_ell_down_backward_dh<<<grid, THREADS_SCATTER, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            grad_output.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(values_down.data_ptr()),
            values_down.stride(0),
            values_down.stride(1),
            col_idx_down.data_ptr<int>(),
            col_idx_down.stride(0),
            reinterpret_cast<__nv_bfloat16*>(d_h.data_ptr()),
            d_h.stride(0),
            N, K_d);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------
    // Kernel 1b: down projection weight gradient → d_W_down
    // Grid (R_model*K_d,), Block THREADS_DW
    // -----------------------------------------------------------------------
    {
        block_ell_down_backward_dw<<<R_model * K_d, THREADS_DW, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),
            h.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            grad_output.stride(0),
            col_idx_down.data_ptr<int>(),
            col_idx_down.stride(0),
            d_values_down_f32.data_ptr<float>(),
            d_values_down_f32.stride(0),
            d_values_down_f32.stride(1),
            N, K_d);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------
    // Kernel 2: SiLU*up backward → d_gate, d_up
    // Grid (ceil(N*d_ff / BLOCK_EW),), Block BLOCK_EW
    // Depends on d_h being fully written (Kernel 1a must finish first).
    // On the same stream, ordering is guaranteed.
    // -----------------------------------------------------------------------
    {
        const int total = N * d_ff;
        const int grid  = (total + BLOCK_EW - 1) / BLOCK_EW;
        silu_mul_backward_kernel<<<grid, BLOCK_EW, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(d_h.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(gate_pre_act.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(up_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(d_gate.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(d_up.data_ptr()),
            total);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------
    // Kernel 3a: gate/up projection backward → d_x scatter
    // Grid (R_ff*(K_g+K_u), ceil(N/BLOCK_N)), Block THREADS_SCATTER
    // -----------------------------------------------------------------------
    {
        const dim3 grid(R_ff * (K_g + K_u), (N + BLOCK_N - 1) / BLOCK_N);
        block_ell_proj_backward_dx<<<grid, THREADS_SCATTER, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(d_gate.data_ptr()),
            d_gate.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(d_up.data_ptr()),
            d_up.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(values_gate.data_ptr()),
            values_gate.stride(0),
            values_gate.stride(1),
            col_idx_gate.data_ptr<int>(),
            col_idx_gate.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(values_up.data_ptr()),
            values_up.stride(0),
            values_up.stride(1),
            col_idx_up.data_ptr<int>(),
            col_idx_up.stride(0),
            reinterpret_cast<__nv_bfloat16*>(d_x.data_ptr()),
            d_x.stride(0),
            N, R_ff, K_g, K_u);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------
    // Kernel 3b: residual backward → d_x += grad_output
    // Grid (ceil(N*d_model / BLOCK_EW),), Block BLOCK_EW
    // -----------------------------------------------------------------------
    {
        const int total = N * d_model;
        const int grid  = (total + BLOCK_EW - 1) / BLOCK_EW;
        residual_add_kernel<<<grid, BLOCK_EW, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(d_x.data_ptr()),
            total);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------
    // Kernel 3c: gate/up weight gradient → d_W_gate, d_W_up
    // Grid (R_ff*(K_g+K_u),), Block THREADS_DW
    // -----------------------------------------------------------------------
    {
        block_ell_proj_backward_dw<<<R_ff * (K_g + K_u), THREADS_DW, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            x.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(d_gate.data_ptr()),
            d_gate.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(d_up.data_ptr()),
            d_up.stride(0),
            col_idx_gate.data_ptr<int>(),
            col_idx_gate.stride(0),
            col_idx_up.data_ptr<int>(),
            col_idx_up.stride(0),
            d_values_gate_f32.data_ptr<float>(),
            d_values_gate_f32.stride(0),
            d_values_gate_f32.stride(1),
            d_values_up_f32.data_ptr<float>(),
            d_values_up_f32.stride(0),
            d_values_up_f32.stride(1),
            N, R_ff, K_g, K_u);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------
    // Cast fp32 weight gradients to bf16 for return
    // -----------------------------------------------------------------------
    return SwiGLUBackwardResult{
        d_x,
        d_values_gate_f32.to(torch::kBFloat16),
        d_values_up_f32.to(torch::kBFloat16),
        d_values_down_f32.to(torch::kBFloat16),
    };
}

} // namespace block_ell
} // namespace titan
