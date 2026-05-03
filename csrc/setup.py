"""
setup.py — Build system for titan_kernels CUDA extension.

Compiles all CUDA kernels and C++ bindings into a single Python extension
module `titan_kernels`.  Loading the module registers all ops under the
`titan::` namespace via TORCH_LIBRARY, making them available as:

    torch.ops.titan.memory_update(...)
    torch.ops.titan.memory_retrieve(...)
    torch.ops.titan.block_ell_swiglu_fwd(...)
    torch.ops.titan.block_ell_swiglu_fwd_with_intermediate(...)
    torch.ops.titan.block_ell_swiglu_bwd(...)
    torch.ops.titan.pin_weights_l2(...)
    torch.ops.titan.unpin_weights_l2()
    torch.ops.titan.persistent_mlp_step(...)

Build:
    cd TPU/csrc
    CUTLASS_PATH=/opt/nvidia/cutlass python setup.py build_ext --inplace

Or via pip:
    pip install -e . --no-build-isolation

Environment variables:
    CUTLASS_PATH  — path to CUTLASS root (default: /opt/nvidia/cutlass)
    TORCH_CUDA_ARCH_LIST — override target arch list (default includes sm_90a)

Target architecture:
    sm_90a is the primary target (Hopper, H100 / DGX Spark).
    RTX 5090 is sm_120 (Blackwell).  sm_90a SASS does not run directly on
    sm_120, but CUDA JIT-compiles the embedded sm_90a PTX for sm_120 at
    first load (forward PTX compatibility).  This works because the kernels
    only use standard WMMA / bf16 intrinsics with no Hopper-only instructions
    (no wgmma, no TMA).  First-load JIT is ~seconds; subsequent loads use the
    cached cubin from ~/.nv/ComputeCache.

    NOTE: do NOT use sm_120 as an nvcc arch flag with current CUTLASS versions —
    CUTLASS sm_120 support is in active development and has known compilation
    bugs.  Use TORCH_CUDA_ARCH_LIST="12.0" only when CUDA 12.8+ is confirmed.
"""

import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# csrc/ root — the directory this setup.py lives in.
CSRC_ROOT = os.path.dirname(os.path.abspath(__file__))

# CUTLASS path (optional; required only if CUTLASS GEMM headers are used).
# Currently our kernels use WMMA (from CUDA toolkit) not CUTLASS templates.
# Set this if you add CUTLASS-based kernels in the future.
CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "/opt/nvidia/cutlass")

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

SOURCES = [
    # Neural memory kernels (memory MLP fused update + retrieve)
    "neural_memory/memory_ops.cpp",
    # memory_kernels.cu: implements the kernel functions declared in memory_kernels.cuh
    # and forward-declared in memory_ops.cpp.
    "neural_memory/memory_kernels.cu",

    # Block-ELL SwiGLU kernels (fused sparse matmul + gate + residual)
    "block_ell/swiglu_fwd.cu",
    "block_ell/swiglu_bwd.cu",
    "block_ell/block_ell_ops.cpp",

    # Loop: L2 cache pinning + persistent MLP step
    "loop/l2_policy.cu",
    "loop/persistent_core.cu",
    "loop/loop_ops.cpp",
]

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------

INCLUDE_DIRS = [
    CSRC_ROOT,                                   # csrc/ root (for cross-folder #include)
    os.path.join(CUTLASS_PATH, "include"),        # CUTLASS main headers
    os.path.join(CUTLASS_PATH, "tools", "util", "include"),  # CUTLASS utility headers
]

# Filter out CUTLASS paths that don't exist (avoids build errors when CUTLASS
# is not installed — only relevant if you add CUTLASS kernels later).
INCLUDE_DIRS = [d for d in INCLUDE_DIRS if os.path.isdir(d) or "cutlass" not in d.lower()]
# Always keep csrc/ root.
INCLUDE_DIRS = [CSRC_ROOT] + [d for d in INCLUDE_DIRS if d != CSRC_ROOT]

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------

# C++17 is required for structured bindings (auto [a, b] = ...).
CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",   # avoids symbol collisions with other extensions
]

# nvcc flags for CUDA device code.
#
# Architecture selection:
#   TORCH_CUDA_ARCH_LIST env var overrides the default arch list (passed to
#   BuildExtension which translates it to -gencode flags).  If not set we use
#   -arch=sm_90a as the default (Hopper / H100 primary target; sm_90a PTX
#   JIT-compiles to sm_120 via CUDA forward compatibility on RTX 5090).
#
#   To build native sm_120 SASS for local 5090 dev (requires CUDA 12.8+):
#     TORCH_CUDA_ARCH_LIST="9.0a 12.0" python setup.py build_ext --inplace
#
_arch_flags: list[str] = []
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    # Default: sm_90a for DGX Spark (H100). PTX forward-compat covers sm_120.
    _arch_flags = ["-arch=sm_90a"]
# else: BuildExtension picks up TORCH_CUDA_ARCH_LIST automatically via
# torch.utils.cpp_extension._get_cuda_arch_flags and injects -gencode flags.

NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    *_arch_flags,
    # Required for __nv_bfloat16 arithmetic (half2 intrinsics, WMMA).
    "--expt-relaxed-constexpr",
    # Required for device-side lambdas in thrust / cooperative groups.
    "--expt-extended-lambda",
    # Enable Tensor Core MMA (WMMA) — used in all three kernel files.
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    # Embed line number info into the binary for CUDA profiler / cuda-gdb.
    "-lineinfo",
    # bf16 math precision — all accumulators are fp32, bf16 only for I/O.
    "-DCUDA_BF16_ENABLE=1",
    # Suppress deprecation warnings from older ATen headers in CUDA 12.x.
    "-Xcudafe", "--diag_suppress=20012",
]

# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------

ext = CUDAExtension(
    name="titan_kernels",
    sources=[os.path.join(CSRC_ROOT, s) for s in SOURCES],
    include_dirs=INCLUDE_DIRS,
    extra_compile_args={
        "cxx":  CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
    # Link against CUDA runtime (automatically added by CUDAExtension, but
    # explicit here for clarity).
    libraries=["cuda", "cudart"],
    # Use RPATH so the extension finds libcuda.so at runtime without LD_LIBRARY_PATH.
    extra_link_args=["-Wl,-rpath,$ORIGIN"],
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup(
    name="titan_kernels",
    version="0.1.0",
    description="Fused CUDA kernels for TitanMAC Looped Block-ELL transformer",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.1.0"],
)
