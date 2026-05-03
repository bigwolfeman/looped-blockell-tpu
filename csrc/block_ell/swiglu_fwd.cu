// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// swiglu_fwd.cu — Fused Block-ELL SwiGLU forward pass implementation.
//
// See swiglu_fwd.cuh for the full algorithm description.
//
// Branch: 006-looped-block-ell
//
// ============================================================================
// DESIGN NOTES
// ============================================================================
//
// Phase 1: block_ell_gate_up_silu_kernel
// ----------------------------------------
// Grid: (R_ff, ceil(N / BLOCK_N))
//   axis-0 = one CTA per output block-row  (16 output features)
//   axis-1 = one CTA per batch tile         (BLOCK_N=32 tokens)
//
// Each CTA:
//   1. Iterates over K_g gate tiles and K_u up tiles.
//   2. For every (gate_k, up_k) pair the CTA needs to load x column-blocks.
//      Because gate and up may have DIFFERENT col_indices, we cannot always
//      share an x tile between the two.  The inner loop therefore handles:
//        a) Accumulate gate:   acc_gate += x[:,col_gate] @ W_gate[r,k]
//        b) Accumulate up:     acc_up   += x[:,col_up]   @ W_up[r,k]
//      Both weight tiles are 16×16 = 256 bf16 values (512 B).
//      The x tile is [BLOCK_N, 16] bf16.
//   3. After all K tiles, compute h = silu(acc_gate) * acc_up in registers.
//   4. Write h [BLOCK_N, 16] to the intermediate buffer.
//
// Tensor Core usage:
//   Each 16×16 tile matmul maps perfectly to one WMMA operation:
//     matrix_a:  x_tile   [BLOCK_N/16, 16] → we tile the batch dimension
//     matrix_b:  W_tile   [16, 16]
//     accum:     fp32     [BLOCK_N/16, 16]
//   With BLOCK_N=32 we get two WMMA calls per tile (2 rows of 16 tokens each).
//   This keeps the tile register pressure low while maintaining Tensor Core use.
//
// Shared memory layout (per CTA):
//   smem_x_gate  [BLOCK_N * 16]   bf16  — x tile for gate (loaded once per k)
//   smem_x_up    [BLOCK_N * 16]   bf16  — x tile for up   (may alias gate if
//                                          col_idx_gate[r,k]==col_idx_up[r,k])
//   smem_w_gate  [16 * 16]        bf16  — weight tile for gate
//   smem_w_up    [16 * 16]        bf16  — weight tile for up
//   Total for BLOCK_N=32, TILE=16:
//     2 * 32*16*2 + 2 * 16*16*2 = 2048 + 1024 = 3072 bytes — well within limits.
//
// Phase 2: block_ell_down_residual_kernel
// -----------------------------------------
// Grid: (R_model, ceil(N / BLOCK_N))
//   Same structure as Phase 1 but input is the intermediate buffer h.
//   In the epilogue: if add_residual, load x[:, r*16:(r+1)*16] and add to acc.
//
// ============================================================================

#include "swiglu_fwd.cuh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// WMMA headers
#include <mma.h>
using namespace nvcuda;

// bf16 arithmetic helpers
#include <cuda_bf16.h>

namespace titan {
namespace block_ell {

// ============================================================================
// Constants and compile-time parameters
// ============================================================================

// Tile size — must match the Block-ELL tile granularity (16×16).
// Changing this requires recompilation; it is a constexpr not a runtime param.
static constexpr int TILE = 16;

// Batch tile size: number of tokens processed per CTA.
// Must be a multiple of TILE (for WMMA) and ≤ 32 for sm_120 shared-memory budget.
static constexpr int BLOCK_N = 32;

// Number of WMMA row-tiles per batch tile: BLOCK_N / TILE = 2.
static constexpr int N_WMMA_ROWS = BLOCK_N / TILE;  // = 2

// ============================================================================
// Shared memory layout helpers
// ============================================================================

// Total shared memory (bytes) needed by Phase 1 CTA:
//   smem_xg    [BLOCK_N * TILE]  bf16  = 32*16*2 = 1024 B
//   smem_xu    [BLOCK_N * TILE]  bf16  = 1024 B
//   smem_wg    [TILE    * TILE]  bf16  = 16*16*2 = 512 B
//   smem_wu    [TILE    * TILE]  bf16  = 512 B
//   epi_gate   [TILE    * TILE]  fp32  = 16*16*4 = 1024 B
//   epi_up     [TILE    * TILE]  fp32  = 1024 B
//   Total = 5120 B
static constexpr int SMEM_PHASE1_BYTES =
    (2 * BLOCK_N * TILE + 2 * TILE * TILE) * sizeof(__nv_bfloat16)
    + 2 * TILE * TILE * sizeof(float);

// Total shared memory (bytes) needed by Phase 2 CTA:
//   smem_h       [BLOCK_N * TILE]  bf16  = 1024 B
//   smem_wd      [TILE    * TILE]  bf16  = 512 B
//   epi_acc_f32  [TILE    * TILE]  fp32  = 1024 B
//   Total = 2560 B
static constexpr int SMEM_PHASE2_BYTES =
    (BLOCK_N * TILE + TILE * TILE) * sizeof(__nv_bfloat16)
    + TILE * TILE * sizeof(float);

// ============================================================================
// Utility: silu  (silu(x) = x * sigmoid(x), computed in fp32)
// ============================================================================
__device__ __forceinline__ float silu_f32(float x) {
    // sigmoid via fast expf; use negf to avoid NaN for large positive x
    return x / (1.0f + expf(-x));
}

// ============================================================================
// Phase 1 kernel: block_ell_gate_up_silu_kernel
//
// Computes h = silu(gate_proj(x)) * up_proj(x) and writes it to buf_h.
// Also writes gate_proj(x) to buf_gate_pre_act and up_proj(x) to buf_up_output
// so the backward pass has the intermediates it needs for the SiLU derivative.
//
// Grid:  (R_ff, ceil(N / BLOCK_N))
// Block: (32,)  — one warp per CTA is sufficient; WMMA needs ≥32 threads.
//        We use 2 warps (64 threads) so that the two N_WMMA_ROWS=2 WMMA ops
//        can be issued by different threads naturally.
//        Actually: WMMA cooperative operations must be called by ALL 32 threads
//        of a warp.  We use 1 warp here (32 threads) and serialise over rows.
// ============================================================================
__global__ void __launch_bounds__(32, 4)
block_ell_gate_up_silu_kernel(
    // Input
    const __nv_bfloat16* __restrict__ x,          // [N, d_model]
    // Gate projection Block-ELL
    const __nv_bfloat16* __restrict__ values_gate, // [R_ff, K_g, TILE, TILE]
    const int32_t*        __restrict__ cidx_gate,  // [R_ff, K_g]
    int32_t K_g,
    // Up projection Block-ELL
    const __nv_bfloat16* __restrict__ values_up,   // [R_ff, K_u, TILE, TILE]
    const int32_t*        __restrict__ cidx_up,    // [R_ff, K_u]
    int32_t K_u,
    // Dimensions
    int32_t N,         // total tokens
    int32_t d_model,   // input feature dim
    int32_t d_ff,      // intermediate (output of this kernel) feature dim
    int32_t R_ff,      // = d_ff / TILE
    // Strides (in elements, not bytes)
    int32_t stride_x_n,      // = d_model (row stride of x)
    int32_t stride_vg_r,     // = K_g * TILE * TILE
    int32_t stride_vg_k,     // = TILE * TILE
    int32_t stride_vu_r,     // = K_u * TILE * TILE
    int32_t stride_vu_k,     // = TILE * TILE
    // Outputs
    __nv_bfloat16* __restrict__ buf_h,             // [N, d_ff]  silu(gate)*up
    __nv_bfloat16* __restrict__ buf_gate_pre_act,  // [N, d_ff]  gate before SiLU (for bwd)
    __nv_bfloat16* __restrict__ buf_up_output      // [N, d_ff]  up projection output (for bwd)
)
{
    // -------------------------------------------------------------------------
    // CTA identity
    // -------------------------------------------------------------------------
    const int pid_r = blockIdx.x;   // which output block-row (0..R_ff-1)
    const int pid_n = blockIdx.y;   // which batch tile   (0..ceil(N/BLOCK_N)-1)

    // Token range for this CTA
    const int tok_base = pid_n * BLOCK_N;
    if (tok_base >= N) return;
    const int tok_end  = min(tok_base + BLOCK_N, N);
    (void)tok_end;  // used implicitly via per-element bounds check below

    // -------------------------------------------------------------------------
    // Shared memory partitioning
    // -------------------------------------------------------------------------
    extern __shared__ __nv_bfloat16 smem[];

    // Layout (in bf16 elements; see host-side smem_bytes_p1 calculation):
    //   smem_xg  [BLOCK_N * TILE] bf16  @ byte 0       (x gather buf for gate)
    //   smem_xu  [BLOCK_N * TILE] bf16  @ byte 1024    (x gather buf for up)
    //   smem_wg  [TILE    * TILE] bf16  @ byte 2048    (weight tile gate)
    //   smem_wu  [TILE    * TILE] bf16  @ byte 2560    (weight tile up)
    //   epi_gate [TILE    * TILE] fp32  @ byte 3072    (epilogue staging gate)
    //   epi_up   [TILE    * TILE] fp32  @ byte 4096    (epilogue staging up)
    //   Total = 5120 bytes
    __nv_bfloat16* smem_xg = smem;
    __nv_bfloat16* smem_xu = smem + BLOCK_N * TILE;
    __nv_bfloat16* smem_wg = smem + 2 * BLOCK_N * TILE;
    __nv_bfloat16* smem_wu = smem + 2 * BLOCK_N * TILE + TILE * TILE;

    // -------------------------------------------------------------------------
    // fp32 accumulators: [N_WMMA_ROWS, TILE] each
    // Stored in register files as WMMA accumulator fragments.
    // N_WMMA_ROWS=2 means we handle 32 tokens (2×16) per CTA.
    // -------------------------------------------------------------------------
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc_gate[N_WMMA_ROWS];
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc_up[N_WMMA_ROWS];

    for (int m = 0; m < N_WMMA_ROWS; ++m) {
        wmma::fill_fragment(acc_gate[m], 0.0f);
        wmma::fill_fragment(acc_up[m],   0.0f);
    }

    // -------------------------------------------------------------------------
    // Base pointers into col_indices for this block-row
    // -------------------------------------------------------------------------
    const int32_t* cidx_gate_row = cidx_gate + pid_r * K_g;
    const int32_t* cidx_up_row   = cidx_up   + pid_r * K_u;

    // -------------------------------------------------------------------------
    // Main accumulation loops
    // -------------------------------------------------------------------------
    // We loop over gate tiles first, then up tiles.  Although we cannot always
    // share x tiles (different col_indices), we load each x tile cooperatively
    // using all 32 threads.
    //
    // Each x tile is [BLOCK_N, TILE] = [32, 16] bf16 = 1024 bytes.
    // 32 threads can load this in a single pass of 32 elements each (32×16/32=16
    // bf16 words per thread = 32 bytes — two 128-bit LDS per thread).
    //
    // Weight tiles are [TILE, TILE] = [16, 16] bf16 = 512 bytes.
    // 32 threads load 16 bf16 each = 32 bytes per thread — one LDS.128.

    const int tid = threadIdx.x;  // 0..31

    // ---- Gate accumulation ----
    for (int k = 0; k < K_g; ++k) {
        const int col = cidx_gate_row[k];

        // Load x tile: x[tok_base : tok_base+BLOCK_N, col*TILE : (col+1)*TILE]
        // Each thread loads BLOCK_N*TILE/32 = 16 elements.
        {
            const int n_elems = BLOCK_N * TILE;
            const __nv_bfloat16* x_tile_src = x + tok_base * stride_x_n + col * TILE;
            // tid loads elements tid, tid+32, tid+64, tid+96, tid+128, tid+160,
            // tid+192, tid+224, tid+256, tid+288, tid+320, tid+352, tid+384,
            // tid+416, tid+448, tid+480  (16 elements per thread)
            #pragma unroll
            for (int e = tid; e < n_elems; e += 32) {
                const int row = e / TILE;  // which token within tile
                const int col_e = e % TILE; // which feature within tile group
                // Bounds check for partial last batch tile
                if (tok_base + row < N) {
                    smem_xg[e] = x_tile_src[row * stride_x_n + col_e];
                } else {
                    smem_xg[e] = __float2bfloat16(0.0f);
                }
            }
        }

        // Load weight tile: values_gate[pid_r, k, :, :]
        // Shape [TILE, TILE] = 256 elements; 32 threads → 8 elements each.
        {
            const __nv_bfloat16* wg_src = values_gate
                + pid_r * stride_vg_r
                + k     * stride_vg_k;
            #pragma unroll
            for (int e = tid; e < TILE * TILE; e += 32) {
                smem_wg[e] = wg_src[e];
            }
        }

        __syncwarp();

        // WMMA: acc_gate[m] += smem_xg[m*TILE : (m+1)*TILE, :] @ smem_wg^T
        // matrix_a is x (row_major  [TILE, TILE])
        // matrix_b is W (col_major  [TILE, TILE]) → matches W stored as [B_out, B_in]
        //   because col_major b means we compute  a @ b^T  = x_block @ W^T
        #pragma unroll
        for (int m = 0; m < N_WMMA_ROWS; ++m) {
            wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __nv_bfloat16, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __nv_bfloat16, wmma::col_major> frag_b;

            wmma::load_matrix_sync(frag_a, smem_xg + m * TILE * TILE, TILE);
            wmma::load_matrix_sync(frag_b, smem_wg, TILE);
            wmma::mma_sync(acc_gate[m], frag_a, frag_b, acc_gate[m]);
        }

        __syncwarp();
    }

    // ---- Up accumulation ----
    for (int k = 0; k < K_u; ++k) {
        const int col = cidx_up_row[k];

        // Load x tile for up projection (may differ from gate col_index)
        {
            const int n_elems = BLOCK_N * TILE;
            const __nv_bfloat16* x_tile_src = x + tok_base * stride_x_n + col * TILE;
            #pragma unroll
            for (int e = tid; e < n_elems; e += 32) {
                const int row = e / TILE;
                const int col_e = e % TILE;
                if (tok_base + row < N) {
                    smem_xu[e] = x_tile_src[row * stride_x_n + col_e];
                } else {
                    smem_xu[e] = __float2bfloat16(0.0f);
                }
            }
        }

        // Load weight tile: values_up[pid_r, k, :, :]
        {
            const __nv_bfloat16* wu_src = values_up
                + pid_r * stride_vu_r
                + k     * stride_vu_k;
            #pragma unroll
            for (int e = tid; e < TILE * TILE; e += 32) {
                smem_wu[e] = wu_src[e];
            }
        }

        __syncwarp();

        #pragma unroll
        for (int m = 0; m < N_WMMA_ROWS; ++m) {
            wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __nv_bfloat16, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __nv_bfloat16, wmma::col_major> frag_b;

            wmma::load_matrix_sync(frag_a, smem_xu + m * TILE * TILE, TILE);
            wmma::load_matrix_sync(frag_b, smem_wu, TILE);
            wmma::mma_sync(acc_up[m], frag_a, frag_b, acc_up[m]);
        }

        __syncwarp();
    }

    // -------------------------------------------------------------------------
    // Epilogue: h = silu(gate_acc) * up_acc, store to buf_h
    // -------------------------------------------------------------------------
    // Each WMMA accumulator fragment holds TILE*TILE fp32 scalars distributed
    // across all 32 threads of the warp (not contiguous in a single register;
    // must be materialised via store_matrix_sync before we can do silu/multiply).
    //
    // We need two [TILE*TILE] fp32 staging areas = 2×1024 = 2048 bytes.
    // The smem layout (established at kernel launch, see host code) reserves
    // explicit fp32 epilogue slots AFTER the bf16 input/weight tiles:
    //
    //   Bytes from smem start:
    //   [0    ) smem_xg  [BLOCK_N * TILE] bf16  = 32*16*2 = 1024 B
    //   [1024 ) smem_xu  [BLOCK_N * TILE] bf16  = 1024 B
    //   [2048 ) smem_wg  [TILE    * TILE] bf16  = 16*16*2 = 512 B
    //   [2560 ) smem_wu  [TILE    * TILE] bf16  = 512 B
    //   [3072 ) epi_gate [TILE    * TILE] fp32  = 16*16*4 = 1024 B
    //   [4096 ) epi_up   [TILE    * TILE] fp32  = 1024 B
    //   Total = 5120 B
    //
    // The smem_xg/xu/wg/wu pointers set up at the top of the kernel are still
    // valid throughout (no aliasing).
    char* smem_bytes    = reinterpret_cast<char*>(smem);
    float* epi_gate_f32 = reinterpret_cast<float*>(smem_bytes
        + (2 * BLOCK_N + 2 * TILE) * TILE * sizeof(__nv_bfloat16));
    float* epi_up_f32   = reinterpret_cast<float*>(smem_bytes
        + (2 * BLOCK_N + 2 * TILE) * TILE * sizeof(__nv_bfloat16)
        + TILE * TILE * sizeof(float));

    // Destination pointers for this block-row strip
    __nv_bfloat16* h_dst    = buf_h            + tok_base * d_ff + pid_r * TILE;
    __nv_bfloat16* gate_dst = buf_gate_pre_act + tok_base * d_ff + pid_r * TILE;
    __nv_bfloat16* up_dst   = buf_up_output    + tok_base * d_ff + pid_r * TILE;

    #pragma unroll
    for (int m = 0; m < N_WMMA_ROWS; ++m) {
        const int tok_m_base = m * TILE;  // token offset within this batch tile

        // Store accumulator fragments to shared memory fp32 staging
        wmma::store_matrix_sync(epi_gate_f32, acc_gate[m], TILE, wmma::mem_row_major);
        wmma::store_matrix_sync(epi_up_f32,   acc_up[m],   TILE, wmma::mem_row_major);
        __syncwarp();

        // Elementwise: h = silu(gate) * up, convert to bf16, store.
        // Also write gate_pre_act and up_output for the backward pass.
        // 32 threads handle TILE*TILE = 256 elements: 8 per thread
        #pragma unroll
        for (int e = tid; e < TILE * TILE; e += 32) {
            const int row = e / TILE;  // row within this WMMA tile (0..TILE-1)
            const int col_e = e % TILE;

            // Global token index
            const int global_tok = tok_base + tok_m_base + row;
            if (global_tok >= N) continue;  // guard last partial tile

            const float g = epi_gate_f32[e];
            const float u = epi_up_f32[e];
            const float h_val = silu_f32(g) * u;

            const int offset = (tok_m_base + row) * d_ff + col_e;
            h_dst[offset]    = __float2bfloat16(h_val);
            gate_dst[offset] = __float2bfloat16(g);   // pre-SiLU gate (for bwd)
            up_dst[offset]   = __float2bfloat16(u);   // up output (for bwd)
        }
        __syncwarp();
    }
}


// ============================================================================
// Phase 2 kernel: block_ell_down_residual_kernel
//
// Computes out = BlockELL_down(h) [+ x if add_residual], writes to result.
//
// Grid:  (R_model, ceil(N / BLOCK_N))
// Block: 32 threads (1 warp)
// ============================================================================
__global__ void __launch_bounds__(32, 4)
block_ell_down_residual_kernel(
    // Intermediate hidden state from Phase 1
    const __nv_bfloat16* __restrict__ buf_h,       // [N, d_ff]
    // Down projection Block-ELL
    const __nv_bfloat16* __restrict__ values_down, // [R_model, K_d, TILE, TILE]
    const int32_t*        __restrict__ cidx_down,  // [R_model, K_d]
    int32_t K_d,
    // Skip-connection input (for residual)
    const __nv_bfloat16* __restrict__ x,           // [N, d_model]
    bool add_residual,
    // Dimensions
    int32_t N,
    int32_t d_ff,
    int32_t d_model,
    int32_t R_model,
    // Strides
    int32_t stride_h_n,     // = d_ff   (row stride of buf_h)
    int32_t stride_x_n,     // = d_model
    int32_t stride_vd_r,    // = K_d * TILE * TILE
    int32_t stride_vd_k,    // = TILE * TILE
    // Output
    __nv_bfloat16* __restrict__ result             // [N, d_model]
)
{
    // -------------------------------------------------------------------------
    // CTA identity
    // -------------------------------------------------------------------------
    const int pid_r = blockIdx.x;
    const int pid_n = blockIdx.y;

    const int tok_base = pid_n * BLOCK_N;
    if (tok_base >= N) return;
    (void)min(tok_base + BLOCK_N, N);  // per-element bounds check used instead

    // -------------------------------------------------------------------------
    // Shared memory
    // Layout:
    //   smem_h     [BLOCK_N * TILE]  bf16   = 1024 B   (h tile gather buffer)
    //   smem_wd    [TILE    * TILE]  bf16   = 512 B    (weight tile)
    //   epi_acc_f32[TILE    * TILE]  fp32   = 1024 B   (epilogue accumulator stage)
    // Total = 2560 B
    // -------------------------------------------------------------------------
    extern __shared__ __nv_bfloat16 smem[];

    __nv_bfloat16* smem_h  = smem;
    __nv_bfloat16* smem_wd = smem + BLOCK_N * TILE;
    float* epi_acc_f32     = reinterpret_cast<float*>(
        reinterpret_cast<char*>(smem) +
        (BLOCK_N + TILE) * TILE * sizeof(__nv_bfloat16)
    );

    // -------------------------------------------------------------------------
    // fp32 accumulator fragments
    // -------------------------------------------------------------------------
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc[N_WMMA_ROWS];
    #pragma unroll
    for (int m = 0; m < N_WMMA_ROWS; ++m) {
        wmma::fill_fragment(acc[m], 0.0f);
    }

    const int32_t* cidx_down_row = cidx_down + pid_r * K_d;
    const int tid = threadIdx.x;

    // -------------------------------------------------------------------------
    // Down projection accumulation
    // -------------------------------------------------------------------------
    for (int k = 0; k < K_d; ++k) {
        const int col = cidx_down_row[k];

        // Load h tile: buf_h[tok_base : tok_end, col*TILE : (col+1)*TILE]
        {
            const int n_elems = BLOCK_N * TILE;
            const __nv_bfloat16* h_src = buf_h + tok_base * stride_h_n + col * TILE;
            #pragma unroll
            for (int e = tid; e < n_elems; e += 32) {
                const int row   = e / TILE;
                const int col_e = e % TILE;
                if (tok_base + row < N) {
                    smem_h[e] = h_src[row * stride_h_n + col_e];
                } else {
                    smem_h[e] = __float2bfloat16(0.0f);
                }
            }
        }

        // Load weight tile: values_down[pid_r, k, :, :]
        {
            const __nv_bfloat16* wd_src = values_down
                + pid_r * stride_vd_r
                + k     * stride_vd_k;
            #pragma unroll
            for (int e = tid; e < TILE * TILE; e += 32) {
                smem_wd[e] = wd_src[e];
            }
        }

        __syncwarp();

        // WMMA: acc[m] += smem_h[m*TILE:(m+1)*TILE, :] @ smem_wd^T
        #pragma unroll
        for (int m = 0; m < N_WMMA_ROWS; ++m) {
            wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __nv_bfloat16, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __nv_bfloat16, wmma::col_major> frag_b;

            wmma::load_matrix_sync(frag_a, smem_h + m * TILE * TILE, TILE);
            wmma::load_matrix_sync(frag_b, smem_wd, TILE);
            wmma::mma_sync(acc[m], frag_a, frag_b, acc[m]);
        }

        __syncwarp();
    }

    // -------------------------------------------------------------------------
    // Epilogue: optionally add residual and store to result
    // -------------------------------------------------------------------------
    // Output base pointer for this block-row: result[:, pid_r*TILE:(pid_r+1)*TILE]
    __nv_bfloat16* out_dst = result + tok_base * d_model + pid_r * TILE;
    // Residual pointer (same layout as result, same block-row strip)
    const __nv_bfloat16* res_src = x + tok_base * stride_x_n + pid_r * TILE;

    #pragma unroll
    for (int m = 0; m < N_WMMA_ROWS; ++m) {
        const int tok_m_base = m * TILE;

        wmma::store_matrix_sync(epi_acc_f32, acc[m], TILE, wmma::mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int e = tid; e < TILE * TILE; e += 32) {
            const int row   = e / TILE;
            const int col_e = e % TILE;
            const int global_tok = tok_base + tok_m_base + row;
            if (global_tok >= N) continue;

            float val = epi_acc_f32[e];

            // Residual add: x[global_tok, pid_r*TILE + col_e]
            if (add_residual) {
                val += __bfloat162float(res_src[(tok_m_base + row) * stride_x_n + col_e]);
            }

            out_dst[(tok_m_base + row) * d_model + col_e] = __float2bfloat16(val);
        }
        __syncwarp();
    }
}


// ============================================================================
// Host function: swiglu_forward_cuda_with_intermediate
// ============================================================================
SwiGLUForwardIntermediates
swiglu_forward_cuda_with_intermediate(
    torch::Tensor x,
    torch::Tensor values_gate,
    torch::Tensor col_idx_gate,
    torch::Tensor values_up,
    torch::Tensor col_idx_up,
    torch::Tensor values_down,
    torch::Tensor col_idx_down,
    bool          add_residual
) {
    // ---- Input validation ----
    TORCH_CHECK(x.is_cuda(),            "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.dim() == 2,           "x must be 2D [N, d_model]");

    TORCH_CHECK(values_gate.is_cuda() && values_up.is_cuda() && values_down.is_cuda(),
                "All value tensors must be on CUDA");
    TORCH_CHECK(values_gate.dtype() == torch::kBFloat16 &&
                values_up.dtype()   == torch::kBFloat16 &&
                values_down.dtype() == torch::kBFloat16,
                "All value tensors must be bf16");

    TORCH_CHECK(col_idx_gate.dtype() == torch::kInt32 &&
                col_idx_up.dtype()   == torch::kInt32 &&
                col_idx_down.dtype() == torch::kInt32,
                "All col_idx tensors must be int32");

    // ---- Geometry ----
    const int64_t N       = x.size(0);
    const int64_t d_model = x.size(1);

    // Gate/Up outputs share d_ff
    const int64_t R_ff  = values_gate.size(0);
    const int64_t K_g   = values_gate.size(1);
    const int64_t K_u   = values_up.size(1);
    const int64_t d_ff  = R_ff * TILE;

    const int64_t R_model = values_down.size(0);
    const int64_t K_d     = values_down.size(1);

    TORCH_CHECK(R_model == d_model / TILE,
        "values_down R_model=", R_model, " inconsistent with d_model=", d_model);
    TORCH_CHECK(values_gate.size(0) == values_up.size(0),
        "gate and up must have same R_ff");
    TORCH_CHECK(values_gate.size(2) == TILE && values_gate.size(3) == TILE,
        "gate tile dims must be ", TILE, "×", TILE);
    TORCH_CHECK(values_down.size(2) == TILE && values_down.size(3) == TILE,
        "down tile dims must be ", TILE, "×", TILE);

    // ---- Make contiguous ----
    auto xc   = x.contiguous();
    auto vg   = values_gate.contiguous();
    auto cig  = col_idx_gate.contiguous();
    auto vu   = values_up.contiguous();
    auto ciu  = col_idx_up.contiguous();
    auto vd   = values_down.contiguous();
    auto cid  = col_idx_down.contiguous();

    // ---- Allocate buffers ----
    auto buf_h            = torch::empty({N, d_ff},    xc.options());
    auto buf_gate_pre_act = torch::empty({N, d_ff},    xc.options());
    auto buf_up_output    = torch::empty({N, d_ff},    xc.options());
    auto result           = torch::empty({N, d_model}, xc.options());

    // ---- CUDA stream ----
    c10::cuda::CUDAGuard device_guard(xc.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ---- Phase 1: gate + up → silu(gate)*up → buf_h ----
    {
        // Shared memory: see SMEM_PHASE1_BYTES constant (5120 bytes).
        const int smem_bytes_p1 = SMEM_PHASE1_BYTES;

        const dim3 grid_p1(
            static_cast<unsigned>(R_ff),
            static_cast<unsigned>((N + BLOCK_N - 1) / BLOCK_N)
        );
        const dim3 block_p1(32);  // 1 warp

        // Strides (in elements)
        const int stride_x_n   = static_cast<int>(xc.stride(0));
        const int stride_vg_r  = static_cast<int>(vg.stride(0));
        const int stride_vg_k  = static_cast<int>(vg.stride(1));
        const int stride_vu_r  = static_cast<int>(vu.stride(0));
        const int stride_vu_k  = static_cast<int>(vu.stride(1));

        block_ell_gate_up_silu_kernel<<<grid_p1, block_p1, smem_bytes_p1, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(vg.data_ptr()),
            cig.data_ptr<int32_t>(),
            static_cast<int32_t>(K_g),
            reinterpret_cast<const __nv_bfloat16*>(vu.data_ptr()),
            ciu.data_ptr<int32_t>(),
            static_cast<int32_t>(K_u),
            static_cast<int32_t>(N),
            static_cast<int32_t>(d_model),
            static_cast<int32_t>(d_ff),
            static_cast<int32_t>(R_ff),
            stride_x_n,
            stride_vg_r, stride_vg_k,
            stride_vu_r, stride_vu_k,
            reinterpret_cast<__nv_bfloat16*>(buf_h.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(buf_gate_pre_act.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(buf_up_output.data_ptr())
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- Phase 2: down(h) + residual → result ----
    {
        // Shared memory: see SMEM_PHASE2_BYTES constant (2560 bytes).
        const int smem_bytes_p2 = SMEM_PHASE2_BYTES;

        const dim3 grid_p2(
            static_cast<unsigned>(R_model),
            static_cast<unsigned>((N + BLOCK_N - 1) / BLOCK_N)
        );
        const dim3 block_p2(32);

        const int stride_h_n  = static_cast<int>(buf_h.stride(0));
        const int stride_x_n  = static_cast<int>(xc.stride(0));
        const int stride_vd_r = static_cast<int>(vd.stride(0));
        const int stride_vd_k = static_cast<int>(vd.stride(1));

        block_ell_down_residual_kernel<<<grid_p2, block_p2, smem_bytes_p2, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(buf_h.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(vd.data_ptr()),
            cid.data_ptr<int32_t>(),
            static_cast<int32_t>(K_d),
            reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
            add_residual,
            static_cast<int32_t>(N),
            static_cast<int32_t>(d_ff),
            static_cast<int32_t>(d_model),
            static_cast<int32_t>(R_model),
            stride_h_n,
            stride_x_n,
            stride_vd_r, stride_vd_k,
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr())
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return SwiGLUForwardIntermediates{result, buf_h, buf_gate_pre_act, buf_up_output};
}


// ============================================================================
// Public API: swiglu_forward_cuda (drops the intermediate buffer)
// ============================================================================
torch::Tensor swiglu_forward_cuda(
    torch::Tensor x,
    torch::Tensor values_gate,
    torch::Tensor col_idx_gate,
    torch::Tensor values_up,
    torch::Tensor col_idx_up,
    torch::Tensor values_down,
    torch::Tensor col_idx_down,
    bool          add_residual
) {
    return swiglu_forward_cuda_with_intermediate(
        x,
        values_gate, col_idx_gate,
        values_up,   col_idx_up,
        values_down, col_idx_down,
        add_residual
    ).result;
}

} // namespace block_ell
} // namespace titan
