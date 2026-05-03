// Copyright 2026 TitanMAC Authors. All rights reserved.
//
// l2_policy.cu — L2 cache persistent-access pinning implementation.
//
// See l2_policy.cuh for full API documentation.
//
// Branch: 006-looped-block-ell

#include "l2_policy.cuh"

#include <cstdint>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

namespace titan {
namespace loop {

// ---------------------------------------------------------------------------
// get_l2_persistent_size
// ---------------------------------------------------------------------------

size_t get_l2_persistent_size() {
    size_t persisting_size = 0;
    // cudaDeviceGetLimit reads the *current* limit, not the hardware cap.
    // The hardware cap can be queried via cudaDeviceGetAttribute with
    // cudaDevAttrL2CacheSize, but the *persisting* budget is separate.
    // We call cudaDeviceGetLimit to get what was last set; if never set it
    // returns 0.  To get the driver-allowed maximum we temporarily probe it.
    C10_CUDA_CHECK(cudaDeviceGetLimit(&persisting_size,
                                      cudaLimitPersistingL2CacheSize));
    if (persisting_size == 0) {
        // Query the hardware L2 size and return ~50% as the practical budget.
        int l2_cache_bytes = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(
            &l2_cache_bytes, cudaDevAttrL2CacheSize, 0));
        // 50% heuristic: matches driver reservation policy on Ampere/Ada/Hopper.
        persisting_size = static_cast<size_t>(l2_cache_bytes) / 2;
    }
    return persisting_size;
}

// ---------------------------------------------------------------------------
// pin_weights_l2
// ---------------------------------------------------------------------------

void pin_weights_l2(
    const std::vector<at::Tensor>& weight_tensors,
    cudaStream_t                   stream
) {
    if (weight_tensors.empty()) {
        return;
    }

    // -----------------------------------------------------------------------
    // Step 1: compute total bytes and find address range.
    // -----------------------------------------------------------------------
    size_t total_bytes = 0;
    uintptr_t min_ptr = std::numeric_limits<uintptr_t>::max();
    uintptr_t max_end = 0;

    for (const auto& t : weight_tensors) {
        TORCH_CHECK(t.is_cuda(),
            "pin_weights_l2: all tensors must be on CUDA, got CPU tensor");
        TORCH_CHECK(t.is_contiguous(),
            "pin_weights_l2: all tensors must be contiguous");

        uintptr_t ptr = reinterpret_cast<uintptr_t>(t.data_ptr());
        size_t    nb  = static_cast<size_t>(t.nbytes());

        total_bytes += nb;
        min_ptr = std::min(min_ptr, ptr);
        max_end = std::max(max_end, ptr + nb);
    }

    // Address-contiguous span (includes gaps between non-adjacent allocations).
    size_t span_bytes = max_end - min_ptr;

    // -----------------------------------------------------------------------
    // Step 2: query device L2 persisting budget and clamp.
    // -----------------------------------------------------------------------
    int l2_cache_bytes = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &l2_cache_bytes, cudaDevAttrL2CacheSize, 0));

    // Driver typically allows up to ~50% of L2 for persisting access.
    // cudaDeviceSetLimit will return cudaErrorInvalidValue if we exceed the cap.
    // Query the current cap by setting and reading back.
    size_t max_persisting = static_cast<size_t>(l2_cache_bytes) / 2;

    size_t reserve_bytes = std::min(span_bytes, max_persisting);
    if (span_bytes > max_persisting) {
        // Print a warning — this is not a hard error; the window will simply
        // cover only the low max_persisting bytes, evicting the rest normally.
        TORCH_WARN("pin_weights_l2: requested ", span_bytes,
                   " bytes but L2 persisting budget is only ", max_persisting,
                   " bytes. Weights may not fully fit in L2.");
        reserve_bytes = max_persisting;
    }

    // -----------------------------------------------------------------------
    // Step 3: reserve persisting L2.
    // -----------------------------------------------------------------------
    C10_CUDA_CHECK(cudaDeviceSetLimit(
        cudaLimitPersistingL2CacheSize, reserve_bytes));

    // -----------------------------------------------------------------------
    // Step 4: attach policy window on the stream covering [min_ptr, max_end).
    //
    // We use a single window spanning the full address range.  Gaps between
    // non-adjacent allocations consume persisting budget wastefully but there
    // is no API to specify multiple windows per stream — only one window is
    // supported.
    //
    // hitRatio=1.0 means 100% of accesses in the window are treated as
    // Persisting (resident in L2 after first use); 0.0 would mean Streaming
    // (evicted immediately after use).  For weights that are reused across
    // every loop iteration, 1.0 is correct.
    // -----------------------------------------------------------------------
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr   = reinterpret_cast<void*>(min_ptr);
    attr.accessPolicyWindow.num_bytes  = span_bytes;
    attr.accessPolicyWindow.hitRatio   = 1.0f;
    attr.accessPolicyWindow.hitProp    = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp   = cudaAccessPropertyStreaming;

    C10_CUDA_CHECK(cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr));
}

// ---------------------------------------------------------------------------
// unpin_weights_l2
// ---------------------------------------------------------------------------

void unpin_weights_l2(cudaStream_t stream) {
    // Clear the policy window by setting num_bytes=0 (or hitRatio=0).
    // CUDA documentation states that setting num_bytes to 0 removes the window.
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio  = 0.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    C10_CUDA_CHECK(cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr));

    // Release the persisting L2 reservation.
    C10_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0));
}

} // namespace loop
} // namespace titan
