# Fused CUDA Kernel Stack — Looped Block-ELL Transformer

**Date**: 2026-05-02
**Target Hardware**: RTX 5090 (sm_120, 192MB L2, 32GB VRAM)
**Toolchain**: CUTLASS 4.2+ / CuTe / C++17 / CUDA 13.0+
**Integration**: PyTorch 2.11+ via `torch.library.custom_op`

---

## 1. Problem Statement

The Looped Block-ELL Transformer runs at 1.5 step/s when all features are active.
The measured baseline without neural memory is 4.0 step/s.
The primary bottleneck is `torch.autograd.grad` inside the neural memory update,
which fragments the torch.compile graph and prevents cross-operator fusion.

Secondary: the core loop (6 blocks × 8 iterations) launches ~300+ individual CUDA
kernels per forward pass. Each kernel launch has overhead and forces intermediate
tensors through HBM unnecessarily.

**Goal**: Recover the 4.0 step/s baseline with all features active, and push
toward 5-6 step/s via persistent kernel weight residency.

---

## 2. Architecture Constraints

### SM_120 Realities
- **No Tensor Memory (TMEM)**, no `tcgen05` MMA instructions
- Must target `sm_90a`-style MMA atoms (Hopper `mma.sync.aligned`)
- CUTLASS 4.2+ required (3.x has no sm_120 support at all)
- Known bugs: grouped GEMM garbage output (needs `compute_120f` + CUDA 13.0),
  FP8 MMA segfault (avoid FP8 MMA on sm_120)
- **Safe path**: target `sm_90a` MMA atoms, which execute correctly on sm_120

### Model Dimensions (pipeline_70k.yaml)
```
d_model     = 512       n_heads     = 8       head_dim    = 64
d_ff        = 1376      n_kv_heads  = 2       kv_groups   = 4  (GQA 4:1)
tile_size   = 16        vocab_size  = 49152   max_seq_len = 1024
n_prelude   = 3         n_core      = 6       n_coda      = 3
mean_depth  = 6         max_depth   = 8       batch_size  = 20
d_memory    = 1024      n_mem_layers= 4       mem_params  ≈ 4.2M
```

### Block-ELL Format
```
values:      [R, K, 16, 16]    bf16    (R = out_dim/16, K = alive tiles per row)
col_indices: [R, K]            int32   (column block index for gather)
```
For SwiGLU at d=512, d_ff=1376:
- w_gate, w_up:  R=86,  C_full=32, K≤32 (d_model → d_ff)
- w_down:        R=32,  C_full=86, K≤86 (d_ff → d_model)

### Compile Graph Topology (Current)
```
[COMPILED GRAPH 1] ─── embed → prelude → retrieve → input_norm → injection
                       → core loop (all iterations) → coda preamble
                       
[GRAPH BREAK]      ─── neural_memory.update() uses autograd.grad
                       (@torch._dynamo.disable)

[COMPILED GRAPH 2] ─── coda remainder → final_norm → lm_head → CE loss
```

---

## 3. Kernel Inventory

### Phase 1 — Eliminate Graph Breaks (Immediate Impact)

#### K1: `neural_memory_update_fused`
**Purpose**: Replace `autograd.grad`-based memory update with a hand-rolled
forward+backward through the 4-layer memory MLP. Eliminates the only hard
graph break in the model.

**What it replaces**:
```python
# Current (causes graph break):
pred = memory_mlp(k)                           # 4-layer MLP forward
loss = F.mse_loss(pred, v)                     # associative loss
grads = torch.autograd.grad(loss, mlp.params)  # <-- graph break
S = eta * S - theta * grads                    # momentum
W = (1 - alpha) * W + S                        # weight update
```

**Fused kernel**:
- Input: `k [B*S, 1024]`, `v [B*S, 1024]`, MLP weights `W_0..W_3` `[1024, 1024]` each,
  momentum buffers `S_0..S_3`, scalars `eta, theta, alpha`
- Output: updated `W_0..W_3` and `S_0..S_3` (in-place), `loss` scalar
- Algorithm: manual forward → MSE loss → manual backward (chain rule through
  4 linear + GELU layers) → momentum → weight update. All in one kernel launch.
- **Implementation**: CUTLASS GEMM for the 1024×1024 matmuls (4 fwd + 4 bwd = 8 GEMMs),
  fused GELU/GELU-grad as EVT epilogue. The 1024×1024 matmuls are small enough that
  a single threadblock can handle each one — pack all 8 into a persistent kernel
  that chains them sequentially.
- **Registration**: `torch.library.custom_op("titan::memory_update", mutates_args=("weights", "momentum"))`

**Expected impact**: Eliminates graph break → torch.compile fuses the entire forward
as a single graph → ~2× speedup from restored fusion alone.

#### K2: `neural_memory_retrieve_fused`
**Purpose**: Fuse the retrieve path: `W_Q` projection → input_norm → 4-layer MLP
forward (under no_grad) → output projection → scale.

**What it replaces**: 7 separate kernel launches (2 GEMMs + 4 MLP GEMMs + norm)

**Fused kernel**:
- Input: `x [B*S, 512]`, `W_Q [512, 1024]`, MLP weights `W_0..W_3`, `W_out [1024, 512]`
- Output: `mem_out [B*S, 512]`
- All read-only on weights (no_grad), pure inference MLP.
- Pack as single CUTLASS grouped-GEMM or chain of GEMM+epilogue.

**Expected impact**: Minor (these are small GEMMs), but eliminates 7 kernel launches.

---

### Phase 2 — Fused Block-ELL SwiGLU (Core MLP Throughput)

#### K3: `block_ell_swiglu_fwd`
**Purpose**: Fuse the three Block-ELL matmuls + SiLU activation + gate multiply +
residual add into a single kernel.

**What it replaces**:
```python
gate = w_gate(x)       # Block-ELL GEMM: [B*S, 512] → [B*S, 1376]
up   = w_up(x)         # Block-ELL GEMM: [B*S, 512] → [B*S, 1376]  (same input!)
h    = silu(gate) * up  # elementwise
out  = w_down(h)        # Block-ELL GEMM: [B*S, 1376] → [B*S, 512]
out  = out + residual   # elementwise
```

**Fusion strategy**:
1. **Gate + Up share input gather**: `x` is gathered once via `col_indices` into
   shared memory. Both gate and up tiles are loaded and matmul'd against the same
   gathered input. Saves 50% of input memory traffic.
2. **SiLU + multiply as epilogue**: After gate and up accumulations complete,
   compute `silu(gate_acc) * up_acc` in registers before storing to shared memory
   for the down projection.
3. **Down + residual as final epilogue**: Down projection accumulates into output,
   adds the residual `x` in the store epilogue.

**Data flow**:
```
Load x_gathered once (shared mem)
  ├─ GEMM: x_gathered × values_gate → gate_acc (registers)
  ├─ GEMM: x_gathered × values_up   → up_acc (registers)
  ├─ Epilogue: silu(gate_acc) * up_acc → h (shared mem or registers)
  ├─ GEMM: h × values_down → out_acc (registers)
  └─ Epilogue: out_acc + residual → store to global
```

**Grid**: `(R_down=32, ceil(B*S/BLOCK_BATCH))` — parallelize over output rows of w_down.
Each threadblock must compute the full d_ff intermediate for its output rows,
which means looping over R_gate=86 rows of gate/up internally.

**Alternative grid**: `(ceil(B*S/BLOCK_BATCH), 1)` with the full SwiGLU pipeline
per threadblock. Each TB handles a batch tile and walks through all block rows.
More registers but better data reuse.

**Sizes**: Input gather: `[BLOCK_BATCH, K, 16]` tiles.
Gate/Up outputs: `[BLOCK_BATCH, 1376]` in shared memory (~86KB for BLOCK_BATCH=32).
This is tight on sm_120 shared memory (100KB default, 228KB max with opt-in).
May need to tile the d_ff dimension and accumulate.

#### K4: `block_ell_swiglu_bwd`
**Purpose**: Fused backward for Block-ELL SwiGLU.

**Computes**:
- `dx`: input gradient [B*S, 512]
- `d_values_gate`, `d_values_up`, `d_values_down`: weight gradients
- Fuses: d_residual passthrough → down backward → silu_grad * up + gate * d_up
  → gate/up backward → scatter to dx

**This is the hardest kernel** — the backward has complex data dependencies between
the three projections. May need to split into 2 kernels:
- K4a: `d_down_bwd` (downstream grad → d_h → d_gate/d_up via silu_grad)
- K4b: `d_gate_up_bwd` (d_gate/d_up → dx via scatter, + dW for gate/up)

---

### Phase 3 — Persistent Core Loop (Maximum Throughput)

#### K5: `persistent_core_loop`
**Purpose**: Keep all core block weights resident in L2 across 8 loop iterations.
Eliminate kernel launch overhead for the entire core loop.

**Architecture**:
```
cudaLaunchCooperativeKernel(persistent_core_loop, ...)

persistent_core_loop:
  // Pin 6 blocks × SwiGLU weights in L2 via cudaAccessPolicyWindow
  // (pre-launch, on the stream)
  
  for t in 0..total_iters:
    h = injection(h, e)              // elementwise, inline
    
    for block in 0..n_core:
      // Attention: use CUTLASS for QKV projection, PyTorch SDPA for attention
      // (SDPA is already fused and optimized — don't rewrite)
      // MLP: K3 (block_ell_swiglu_fwd) inline
      
    cooperative_groups::this_grid().sync()  // device-wide barrier
    
    h = where(active, h_new, h)     // elementwise, inline
```

**Challenge**: Attention (SDPA) can't easily be inlined into a persistent kernel
because PyTorch's SDPA dispatcher selects the optimal backend at runtime.
Two options:
  (a) Call SDPA as a sub-kernel from within the persistent kernel (not possible
      with cooperative launch — can't launch child kernels)
  (b) Implement attention ourselves using CUTLASS grouped-GEMM for QKV + a
      custom fused attention kernel

**Realistic approach**: Make the persistent kernel cover the MLP path only
(injection → norm → SwiGLU → residual), and let attention remain as separate
SDPA calls between iterations. This still captures the biggest win (weight
residency for the 3 SwiGLU matrices per block × 6 blocks = 18 weight tensors).

**Weight budget for L2 residency (d=512, d_ff=1376, 25% density)**:
```
Per core block (post-compact, K ≈ 0.25 * C_full):
  w_gate: 86 × 8 × 16 × 16 × 2B  = 0.34 MB
  w_up:   86 × 8 × 16 × 16 × 2B  = 0.34 MB
  w_down: 32 × 22 × 16 × 16 × 2B = 0.34 MB
  col_indices: ~1 KB
  Subtotal: ~1.0 MB per block
  
6 core blocks: 6.1 MB total

Attention weights per block (dense):
  QKV (GQA): 512 × (512 + 2×128) × 2B = 0.75 MB
  out_proj:  512 × 512 × 2B = 0.5 MB
  Subtotal: 1.25 MB per block
  
6 core blocks attention: 7.5 MB

Total core weights: ~13.6 MB  (fits easily in 192MB L2)
```

Even pre-compact (100% density), total is ~38 MB — still fits in L2.

---

### Phase 4 — Fused Attention + Norm (Optional, Diminishing Returns)

#### K6: `fused_norm_qkv_rope`
Fuse RMSNorm → Q/K/V projection → RoPE → QK-Norm into one kernel.
Eliminates 5 kernel launches and 4 intermediate tensor materializations per block.

#### K7: `fused_out_proj_residual_norm`
Fuse attention output projection → residual add → pre-MLP RMSNorm.
Eliminates the gap between attention and MLP.

These are standard Transformer fusion targets (FlashAttention-3 style).
Lower priority because attention is already well-optimized by SDPA.

---

## 4. Implementation Plan

### Phase 1: Graph Break Elimination (1-2 weeks)
```
csrc/
  neural_memory/
    memory_update.cu      -- K1: fused update kernel
    memory_retrieve.cu    -- K2: fused retrieve kernel  
    memory_ops.cpp        -- torch.library registrations + backward
  CMakeLists.txt
  setup.py               -- CUDAExtension build
```

**Deliverable**: Neural memory as opaque compiled op. Single torch.compile graph
for entire forward pass. Measure step/s improvement.

**Build**:
```bash
pip install -e csrc/
# or
python csrc/setup.py build_ext --inplace
```

### Phase 2: Fused Block-ELL SwiGLU (2-3 weeks)
```
csrc/
  block_ell/
    swiglu_fwd.cu         -- K3: fused forward
    swiglu_bwd.cu         -- K4: fused backward (may be 2 kernels)
    block_ell_ops.cpp      -- torch.library registrations
```

**Deliverable**: Drop-in replacement for PrunableLinear in SwiGLU MLPs.
Test: numerical equivalence with existing einsum path (atol=1e-2 for bf16).

### Phase 3: Persistent Loop (2-3 weeks)
```
csrc/
  loop/
    persistent_core.cu     -- K5: cooperative launch persistent kernel
    l2_policy.cu           -- cudaAccessPolicyWindow helpers
    loop_ops.cpp           -- torch.library registration
```

**Deliverable**: Entire core loop (MLP path) as one kernel launch with L2
weight residency. Attention still calls SDPA between iterations.

### Phase 4: Attention Fusion (optional, 1-2 weeks)
Standard norm+projection fusion. Low priority.

---

## 5. Build System

```python
# csrc/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

CUTLASS = os.environ.get("CUTLASS_PATH", "/opt/nvidia/cutlass")

setup(
    name="titan_kernels",
    ext_modules=[CUDAExtension(
        name="titan_kernels",
        sources=[
            "neural_memory/memory_update.cu",
            "neural_memory/memory_retrieve.cu",
            "neural_memory/memory_ops.cpp",
            "block_ell/swiglu_fwd.cu",
            "block_ell/swiglu_bwd.cu",
            "block_ell/block_ell_ops.cpp",
            "loop/persistent_core.cu",
            "loop/l2_policy.cu",
            "loop/loop_ops.cpp",
        ],
        include_dirs=[
            f"{CUTLASS}/include",
            f"{CUTLASS}/tools/util/include",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3", "-std=c++17",
                "-arch=sm_90a",               # Hopper MMA atoms (safe on sm_120)
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            ],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
```

**Dependencies**:
- CUDA 13.0+ (sm_120 + compute_120f for CUTLASS bug workaround)
- CUTLASS 4.2+ (sm_120 support)
- PyTorch 2.11+ (torch.library.custom_op API)
- CuTe (bundled with CUTLASS 4.x)

---

## 6. PyTorch Integration Pattern

### torch.library.custom_op (per kernel)

```cpp
// memory_ops.cpp
#include <torch/library.h>

torch::Tensor memory_update_impl(
    torch::Tensor keys,        // [B*S, d_memory]
    torch::Tensor values,      // [B*S, d_memory]
    torch::Tensor weights,     // [n_layers, d_memory, d_memory] (mutated)
    torch::Tensor momentum,    // [n_layers, d_memory, d_memory] (mutated)
    double eta, double theta, double alpha
);

// Shape inference for torch.compile
torch::Tensor memory_update_fake(
    torch::Tensor keys, torch::Tensor values,
    torch::Tensor weights, torch::Tensor momentum,
    double eta, double theta, double alpha
) {
    return torch::empty({1}, keys.options());  // scalar loss
}

TORCH_LIBRARY(titan, m) {
    m.def("memory_update(Tensor keys, Tensor values, Tensor(a!) weights, "
          "Tensor(b!) momentum, float eta, float theta, float alpha) -> Tensor");
    m.impl("memory_update", torch::kCUDA, memory_update_impl);
    m.impl("memory_update", torch::kMeta, memory_update_fake);  // FakeTensor
}
```

```python
# Python-side usage (drop-in replacement in neural_memory.py)
import titan_kernels  # loads the TORCH_LIBRARY

class NeuralMemory(nn.Module):
    def update(self, h, ...):
        k = self.W_K(h)
        v = self.W_V(h)
        loss = torch.ops.titan.memory_update(
            k.reshape(-1, self.d_memory),
            v.reshape(-1, self.d_memory),
            self.mlp_weights,      # mutated in-place
            self.momentum_buffer,  # mutated in-place
            self.eta, self.theta, self.alpha,
        )
        return loss
```

---

## 7. Testing Strategy

### Numerical Equivalence Tests
For each kernel, compare against the existing PyTorch/Triton implementation:
- K1: `memory_update_fused` vs current `autograd.grad` path (fp32 ref, bf16 kernel)
- K3: `block_ell_swiglu_fwd` vs sequential `w_gate → w_up → silu*up → w_down → residual`
- K4: backward equivalence via `torch.autograd.gradcheck` against PyTorch reference

Tolerances: `atol=1e-2, rtol=1e-2` for bf16 (tile matmuls accumulate in fp32 internally).

### Performance Tests
- Kernel-level: nsys profile each kernel, compare FLOP/s and memory bandwidth vs roofline
- End-to-end: step/s measurement at each phase:
  - Baseline (current): 1.5 step/s
  - After Phase 1 (graph break fix): target 3.5-4.0 step/s
  - After Phase 2 (fused SwiGLU): target 4.5-5.0 step/s
  - After Phase 3 (persistent loop): target 5.5-6.5 step/s

### Correctness Tests
- Train for 1000 steps with fused kernels, compare loss curve against reference
- Gradient accumulation test: 10 steps, compare parameter state vs reference
- Prune → compact → fused kernel transition test

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CUTLASS sm_120 bugs | High | Target sm_90a atoms; test with compute_120f flag |
| K3 shared memory overflow (86KB SwiGLU intermediate) | Medium | Tile d_ff dimension; use 228KB opt-in via cudaFuncSetAttribute |
| Persistent kernel occupancy limits | Medium | Profile max active blocks; may need to reduce BLOCK_BATCH |
| K4 backward complexity | High | Accept 2-kernel split; focus on K3 forward first |
| CuTe learning curve | Medium | Use CuTe for layout only, not for control flow |

---

## 9. Reference Materials

- CUTLASS 4.x: github.com/NVIDIA/cutlass (examples 47, 49, 83)
- Persistent RNNs: github.com/baidu-research/persistent-rnn (Diamos 2016)
- Mirage MPK: github.com/mirage-project/mirage (full-model megakernel)
- CUTLASS Ping-Pong: pytorch.org/blog/cutlass-ping-pong-gemm-kernel
- L2 Cache Control: docs.nvidia.com/cuda/cuda-programming-guide/l2-cache-control
- EVT Epilogue Fusion: research.colfax-intl.com/epilogue_visitor_tree
- torch.library API: pytorch.org/docs/stable/library.html
