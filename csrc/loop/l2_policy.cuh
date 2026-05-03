// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// l2_policy.cuh — L2 cache persistent-access pinning helpers for RTX 5090.
//
// The RTX 5090 (sm_120) has 192MB of L2 cache.  After compaction the core
// loop's weight tensors total ~13.6MB, which fits well within the persisting
// window (typically capped at ~50% of L2 by the driver, i.e. ~96MB).
//
// How CUDA L2 persistence works:
//   cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)
//     — reserves `bytes` of L2 exclusively for streams that use
//       cudaAccessPropertyPersisting.
//
//   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)
//     — attaches a policy window to a stream so that every load issued
//       from that stream within [base_ptr, base_ptr+num_bytes) gets
//       hitProp treatment (Persisting) and all other loads get
//       missProp (Streaming).
//
// Usage pattern:
//   1. Call pin_weights_l2(tensors, stream) once before the training loop.
//   2. Launch all core-loop CUDA kernels on the same stream.
//   3. At the end of training (or when weights change significantly due to
//      topology update / compaction) call unpin_weights_l2(stream).
//
// Notes:
//   - Only one policy window is active per stream at a time.  Calling
//     pin_weights_l2 again replaces the previous window.
//   - Multiple tensors → set the window to cover the largest contiguous
//     address range.  Non-contiguous gaps waste persisting budget but are
//     harmless.
//   - The persisting size limit is device-global.  Multiple concurrent
//     streams each reserving persisting access will share the budget
//     round-robin per SM.
//
// Namespace: titan::loop
// Target:    sm_90a (compatible with sm_120)
//
// Branch: 006-looped-block-ell

#pragma once

#include <cstddef>
#include <vector>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace titan {
namespace loop {

// ---------------------------------------------------------------------------
// pin_weights_l2
//
// Reserves L2 persistent cache space and attaches a policy window on `stream`
// covering all tensors in `weight_tensors`.
//
// Algorithm:
//   1. Sum nbytes across all tensors to get total_bytes.
//   2. Query max persisting L2 size via cudaDeviceGetLimit.
//   3. Clamp total_bytes to that limit (warn if clamped).
//   4. Call cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, total_bytes).
//   5. Find the contiguous address range [min_ptr, max_ptr+max_nbytes).
//   6. Attach that range as a single cudaAccessPolicyWindow with
//      hitProp=Persisting, missProp=Streaming, hitRatio=1.0.
//
// Arguments:
//   weight_tensors  — list of bf16/fp32 CUDA tensors to pin (must be on same
//                     device as stream)
//   stream          — CUDA stream to attach the policy window to
//
// Throws c10::Error if any tensor is not on CUDA or CUDA API calls fail.
// ---------------------------------------------------------------------------
void pin_weights_l2(
    const std::vector<at::Tensor>& weight_tensors,
    cudaStream_t                   stream
);

// ---------------------------------------------------------------------------
// unpin_weights_l2
//
// Detaches the L2 persistent-access policy window from `stream` and releases
// the persisting L2 reservation back to the driver.
//
// Call this when:
//   - Topology update / compaction changes the weight tensor addresses.
//   - The training loop ends.
//   - You want to free persisting budget for another workload.
// ---------------------------------------------------------------------------
void unpin_weights_l2(cudaStream_t stream);

// ---------------------------------------------------------------------------
// get_l2_persistent_size
//
// Returns the maximum bytes that the current device allows to be reserved
// as persistent L2.  Typically ~50% of total L2 cache size.
//
// Use this to check if your weights fit before calling pin_weights_l2.
// ---------------------------------------------------------------------------
size_t get_l2_persistent_size();

} // namespace loop
} // namespace titan
