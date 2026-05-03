# Looped Block-ELL Transformer — Architecture Diagram

## Overview

A Parcae-style looped transformer with Block-ELL sparse MLPs, neural long-term
memory, compressed sparse attention, and iteration-aware ReMoE routing.

**Design goal**: Match a 34-layer dense transformer's quality with 6 unique
layers looped 8x at 25% MLP density.

```
                         Looped Block-ELL Transformer
    ================================================================

    tokens ──► [Hybrid Embedding] ──► x_0 ∈ R^{B×S×d}
                     │
                     │   Euclidean + Lorentz (hyperbolic)
                     │   x = [x_euc ; x_lor]  where x_lor ∈ H^n
                     │   scale: x = x · √d
                     ▼
              ┌─────────────┐
              │  PRELUDE    │  3 TransformerBlocks (non-looped, dense)
              │  Block 0    │  Each: RMSNorm → MHA → RMSNorm → SwiGLU
              │  Block 1    │
              │  Block 2    │
              └──────┬──────┘
                     │
                     ├──────────────────────────────────┐
                     │                                  │
                     ▼                                  ▼
         ┌──── Input Norm ────┐            ┌── Neural Memory ──┐
         │  e = RMSNorm(x)    │            │  Retrieve ONCE    │
         └────────┬───────────┘            │  before core loop │
                  │                        └────────┬──────────┘
                  │                                 │
                  ▼                                 │ mem_out
    h_0 ~ N(0, 0.02²)  (random init)               │
                  │                                 │
    ══════════════╪═════════════════════════════════╪══════
    ║  CORE LOOP  ║  T iterations (T ~ Poisson(6)) ║     ║
    ║             ║  shared weights across iters    ║     ║
    ║             ▼                                 ▼     ║
    ║  ┌────────────────────────────────────────────────┐ ║
    ║  │                                                │ ║
    ║  │  ┌─── Diagonal Injection (SSM-style) ───────┐  │ ║
    ║  │  │                                          │  │ ║
    ║  │  │  A = -exp(log_A)          (negative)     │  │ ║
    ║  │  │  dt = softplus(log_dt)    (positive)     │  │ ║
    ║  │  │  decay = exp(dt · A)      ∈ (0, 1)      │  │ ║
    ║  │  │                                          │  │ ║
    ║  │  │  h_new = decay · h + dt · e              │  │ ║
    ║  │  │                                          │  │ ║
    ║  │  │  Spectral radius < 1 by construction     │  │ ║
    ║  │  │  (ZOH discretization of stable ODE)      │  │ ║
    ║  │  └──────────────────────────────────────────┘  │ ║
    ║  │                    │                            │ ║
    ║  │                    ▼                            │ ║
    ║  │  ┌─── Memory Residual ──────────────────────┐  │ ║
    ║  │  │  h_new = h_new + γ · Norm(Proj(mem_out)) │  │ ║
    ║  │  │  γ: learned scale (zero-init)            │  │ ║
    ║  │  └──────────────────────────────────────────┘  │ ║
    ║  │                    │                            │ ║
    ║  │                    ▼                            │ ║
    ║  │  ┌─── Core Block 0 ─────────────────────────┐  │ ║
    ║  │  │  (repeated for blocks 0..5)               │  │ ║
    ║  │  │                                           │  │ ║
    ║  │  │  ┌─ Multi-Head Attention ──────────────┐  │  │ ║
    ║  │  │  │  Pre-norm: x' = RMSNorm(x)          │  │  │ ║
    ║  │  │  │                                     │  │  │ ║
    ║  │  │  │  Q = W_Q · x'    [n_heads × d_h]   │  │  │ ║
    ║  │  │  │  K = W_KV · x'   [n_kv × d_h] GQA  │  │  │ ║
    ║  │  │  │  V = W_KV · x'   [n_kv × d_h]      │  │  │ ║
    ║  │  │  │                                     │  │  │ ║
    ║  │  │  │  Q = QKNorm(Q),  K = QKNorm(K)      │  │  │ ║
    ║  │  │  │  Q = RoPE(Q),    K = RoPE(K)        │  │  │ ║
    ║  │  │  │  K,V = expand(K,V, groups=4)        │  │  │ ║
    ║  │  │  │                                     │  │  │ ║
    ║  │  │  │  ┌─ if S ≤ W: SDPA ──────────────┐ │  │  │ ║
    ║  │  │  │  │  out = SDPA(Q,K,V, causal)     │ │  │  │ ║
    ║  │  │  │  └────────────────────────────────┘ │  │  │ ║
    ║  │  │  │  ┌─ if S > W: CSA ────────────────┐ │  │  │ ║
    ║  │  │  │  │  K_c,V_c = Compress(x, GQA)    │ │  │  │ ║
    ║  │  │  │  │  scores = [compressed|window|   │ │  │  │ ║
    ║  │  │  │  │           sink]                 │ │  │  │ ║
    ║  │  │  │  │  out = softmax(scores) · V_all  │ │  │  │ ║
    ║  │  │  │  └────────────────────────────────┘ │  │  │ ║
    ║  │  │  │                                     │  │  │ ║
    ║  │  │  │  XSA: out -= proj_v(out, V)         │  │  │ ║
    ║  │  │  │  out = W_O · out                    │  │  │ ║
    ║  │  │  └─────────────────────────────────────┘  │  │ ║
    ║  │  │  x = x + attn_out       (residual)        │  │ ║
    ║  │  │                                           │  │ ║
    ║  │  │  ┌─ SwiGLU MLP (Block-ELL Sparse) ─────┐ │  │ ║
    ║  │  │  │  Pre-norm: x' = RMSNorm(x)          │ │  │ ║
    ║  │  │  │                                      │ │  │ ║
    ║  │  │  │  gate = W_gate · x'   [d_ff]        │ │  │ ║
    ║  │  │  │  up   = W_up   · x'   [d_ff]        │ │  │ ║
    ║  │  │  │  out  = W_down · (SiLU(gate) * up)   │ │  │ ║
    ║  │  │  │                                      │ │  │ ║
    ║  │  │  │  W_gate, W_up, W_down are            │ │  │ ║
    ║  │  │  │  PrunableLinear (16×16 tiles)         │ │  │ ║
    ║  │  │  │  Pruned to ~25% density via CMS      │ │  │ ║
    ║  │  │  │  Post-compact: Block-ELL format       │ │  │ ║
    ║  │  │  └──────────────────────────────────────┘ │  │ ║
    ║  │  │  x = x + mlp_out        (residual)        │  │ ║
    ║  │  └───────────────────────────────────────────┘  │ ║
    ║  │                    │                             │ ║
    ║  │                    ▼                             │ ║
    ║  │  ┌─── Depth Control ────────────────────────┐   │ ║
    ║  │  │  if t < n_max: h_new = h_new.detach()    │   │ ║
    ║  │  │  active = (t < depth_i)                  │   │ ║
    ║  │  │  h = where(active, h_new, h)  (freeze)   │   │ ║
    ║  │  └──────────────────────────────────────────┘   │ ║
    ║  │                    │                             │ ║
    ║  └────────────────────┼─────────────────────── t++ │ ║
    ║                       │                             ║
    ══════════════════════╪═══════════════════════════════
                          │
                          ▼
              ┌──── Neural Memory Update ────┐
              │  (every N steps only)        │
              │  See below for details       │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌─────────────┐
              │  CODA       │  3 TransformerBlocks (non-looped, dense)
              │  Block 0    │  Same architecture as prelude
              │  Block 1    │
              │  Block 2    │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Final Norm  │  RMSNorm
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   LM Head   │  Weight-tied: logits = x · E^T
              │  (chunked)  │  CE computed in chunks to save VRAM
              └──────┬──────┘
                     │
                     ▼
                   loss
```

---

## Component Details

### 1. Hybrid Embedding

```
x = [Embedding_euc(token) ; LorentzLift(Embedding_lor(token))]
```

- **Euclidean part** (d_euc = d/2): Standard learned embedding
- **Lorentz part** (d_lor = d/2): Learned embedding lifted onto hyperboloid H^n
  via `x_0 = sqrt(1 + ||x_{1:}||^2)`, preserving Minkowski inner product
- **Why**: Hyperbolic geometry naturally encodes hierarchical/tree structures
  (syntax trees, taxonomies). Euclidean handles flat semantic similarity.

### 2. Prelude & Coda (Non-Looped Blocks)

Standard pre-norm transformer blocks. **Not pruned** — full dense compute.

```
x = x + MHA(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

- **Prelude** (3 blocks): Builds initial representation from token embeddings.
  Operates before the recurrent core loop.
- **Coda** (3 blocks): Refines the looped representation into prediction-ready
  features. Operates after the core loop completes.

### 3. Diagonal Injection (SSM-Style Stability Gate)

```
A = -exp(log_A)           # Learned, constrained negative
dt = softplus(log_dt)      # Learned, constrained positive
decay = exp(dt * A)        # Per-dimension, ∈ (0, 1)

h_new = decay * h + dt * e
```

- **Guarantees spectral radius < 1** by construction: `decay = exp(negative) < 1`
- Based on Zero-Order Hold (ZOH) discretization of the continuous ODE: `dh/dt = Ah + Be`
- `e` is the input norm output (fixed across iterations); `h` is the recurrent state
- **Init**: `decay ≈ 0.447` (geometric mean of useful range)
- Prevents gradient explosion across loop iterations without clipping

### 4. Core Loop (Parcae-Style Shared Weights)

```
for t in 0..T-1:                           # T ~ Poisson(mean=6), max=8
    h = injection(h, e)                     # SSM gate
    h = h + memory_residual                 # Neural memory (if active)
    for block in core_blocks:               # 6 shared-weight blocks
        h = block(h)                        # Attention + SwiGLU
    if t < n_max: h = stop_gradient(h)      # Truncated BPTT
    h = where(t < depth_i, h_new, h_old)    # Per-sequence freeze
```

**Key properties:**
- **Weight sharing**: 6 unique blocks reused across T iterations = effective
  depth of 3 + 6×T + 3 layers with only 12 unique layers of parameters
- **Poisson depth**: Each sequence samples `T_i ~ Poisson(6)`, clamped to [1, 8].
  Finished sequences are *frozen* via `where()`, not zeroed.
- **Truncated BPTT**: Gradients flow through only the last `k_max` iterations.
  Earlier iterations run in `stop_gradient` mode (no-grad forward only).
- **Activation checkpointing**: Core body is wrapped in `checkpoint()` — 
  activations recomputed during backward to save VRAM.

### 5. Multi-Head Attention (GQA + QK-Norm + CSA + XSA)

#### Grouped Query Attention (GQA)

```
Q = W_Q · x        ∈ R^{B×S×n_heads×d_h}     # 8 query heads
[K; V] = W_KV · x  ∈ R^{B×S×n_kv×d_h}        # 2 KV heads (4:1 ratio)
K = repeat(K, groups=4)                         # Expand to match Q
V = repeat(V, groups=4)
```

- 4 query heads share each KV head → 75% KV parameter reduction
- Proven equivalent quality to full MHA at this scale (DeepSeek, Llama 2/3)

#### QK-Norm (Gemma 2 Style)

```
Q = RMSNorm_Q(Q)       # Separate learned scale per head_dim
K = RMSNorm_K(K)       # Applied BEFORE RoPE
```

- Prevents attention logit growth as model trains longer
- Separate norm parameters for Q and K (not shared)

#### Compressed Sparse Attention (CSA, DeepSeek-V2 Style)

Activated when sequence length exceeds sliding window:

```
# Compress KV cache
x_c = AvgPool1D(x, ratio=8, stride=4)     # [B, S/4, d]
K_c, V_c = W_KV_compressed(x_c)            # GQA dimensions

# Three attention branches with single softmax:
scores = [
    Q · K_c^T / √d_h,     # Compressed global (causal masked)
    Q · K^T / √d_h,       # Sliding window (width W)
    sink_logit             # Learned attention sink
]
weights = softmax(concat(scores))
out = weights_c · V_c + weights_w · V
```

- **Window**: 4096 tokens (at seq_len ≤ 4096, falls back to full SDPA)
- **Compression**: 8:4 ratio → each compressed entry covers 8 tokens, stride 4
- **Sink**: Learnable per-head scalar, always attended to (StreamingLLM insight)

#### Exclusive Self-Attention (XSA, arXiv:2603.09078)

```
# After attention weighted sum:
proj = (out · V) / (V · V)        # Scalar projection coefficient
out = out - proj * V               # Remove self-value component
```

- Prevents attention from "attending to itself" — removes the trivial solution
  where a token copies its own value vector
- Forces attention to learn cross-token relationships
- ~1-2% PPL improvement, zero parameters

### 6. SwiGLU MLP (Block-ELL Sparse)

```
gate = W_gate · x                  # [B, S, d_ff]     d_ff = (8/3)·d
up   = W_up   · x                  # [B, S, d_ff]
out  = W_down · (SiLU(gate) * up)  # [B, S, d_model]

SiLU(x) = x · sigmoid(x)
```

- **SwiGLU** (LLaMA-style): Gated linear unit with SiLU activation.
  Two projections (gate, up) multiplied element-wise after activation.
  d_ff = (8/3)×d rounded to tile_size multiple = 1376 for d=512.
- **Block-ELL Sparse**: W_gate, W_up, W_down stored as 16×16 tiles.
  During pruning, lowest-gradient tiles are killed.
  After compaction, stored as [R, K_new, 16, 16] + col_indices.

#### CMS Gradient-Based Pruning

```
# Between loss.backward() and optimizer.step():
for each PrunableLinear:
    grad_tiles = weight.grad.reshape(R, 16, C, 16)
    tile_norms = ||grad_tiles||_F           # Frobenius norm per tile
    score_ema = 0.95 * score_ema + 0.05 * tile_norms

# Every prune_interval steps:
    kill bottom 10% of alive tiles by score
    density *= 0.9
```

- 13 pruning rounds over 40k steps → 0.9^13 ≈ 25% final density
- Only core block MLPs are pruned (prelude/coda stay dense)

#### Block-ELL Compaction

```
# At prune_end (step 50k):
K_new = min(alive tiles per row across all rows)
values = gather top-K_new tiles per row → [R, K_new, 16, 16]
col_indices = which input columns each tile maps to → [R, K_new]
```

Forward after compaction:
```
x_blocks = x.reshape(B, S, C, 16)
x_gathered = x_blocks[:, :, col_indices, :]     # [B, S, R, K_new, 16]
out = einsum("bsrki, rkoi -> bsrko", x_gathered, values)
out = out.sum(dim=K)                             # [B, S, R, 16]
out = out.reshape(B, S, out_features)
```

### 7. Neural Long-Term Memory (Titans, arXiv:2501.00663)

The memory M is a **deep MLP whose weights ARE the memory**. Updated via
gradient descent on an associative loss at each training step.

```
# Retrieve (before core loop, once per forward pass):
q = W_Q · x                               # Query projection
mem_out = stop_grad(MLP_M(q))              # Forward through memory MLP
                                           # (stop-grad on MLP weights only)

# Residual injection (every loop iteration):
h = h + γ · RMSNorm(W_proj(mem_out))      # γ: learned scale, zero-init

# Update (after core loop, every N steps):
k = W_K · h_final                          # Key projection
v = W_V · h_final                          # Value projection
l(M; x) = ||M(k) - v||²                   # Associative loss (Eq. 12)
∇_M = ∂l/∂M                               # Gradient w.r.t. MLP weights

# Momentum update (Eq. 13):
S_t = η · S_{t-1} - θ · clip(∇_M)         # η = 0.95 (fixed momentum decay)
                                           # θ = 0.01 (memory learning rate)

# Weight update with surprise-modulated forgetting (Eq. 14):
surprise = l_t / EMA(l)                    # How surprising is this input?
α = sigmoid(3.0 · (surprise - 1))          # Map to [α_min, α_max]
α = α_min + α · (α_max - α_min)           # α_min=0.0001, α_max=0.003

M_t = (1 - α) · M_{t-1} + S_t             # Forget + momentum update
```

**Key insight**: The memory is an SSM where the state is the MLP weights
themselves — a continuous compression of all past tokens. The surprise gate
self-regulates: boring inputs → low α → retain memory; surprising inputs →
high α → partially forget and rewrite.

- **4 layers**, d_memory=1024, ~4.2M parameters
- **Update every 5 steps** to amortize the forward+backward cost
- **Warmup**: Off for 1k steps, linear ramp 0→1 over steps 1k-5k
- Memory MLP excluded from outer optimizer (updated only by inner loop)

### 8. ReMoE Routing (Post-Compaction, Phase C)

After compaction at step 50k, tile-groups in the Block-ELL layers are
organized into clusters. A ReMoE router selects which clusters to activate
per token:

```
# Router: x → ReLU gates per cluster
gates = ReLU(W_2 · ReLU(W_1 · x))         # [B, S, n_clusters]

# Gated forward: scale cluster outputs by gate values
out_c = Σ_c gates_c · BlockELL_c(x)       # Per-cluster sparse matmul

# L1 regularization drives sparsity:
L_route = Σ_c λ_c · frac_active_c · mean_gate_c

# Adaptive λ per cluster (zeroth-order):
λ_c ← clip(λ_c · 1.2^sign(sparsity_c - target), 0.01, 100)
```

- **16 clusters** of contiguous tile-rows
- **Target 50% sparsity** → ~12.5% net FLOPs (25% density × 50% routing)
- **Warmup**: 5k steps, gradually enabling routing
- Different loop iterations activate different tile groups (iteration-aware)

### 9. Attention Residuals (JAX Only — Scan Carry)

Depth-wise attention over all loop iteration outputs, aggregated before coda.
**Properly implemented in JAX via scan carry** (O(1) memory, full gradients):

```
# In scan carry state:
(h, acc, lse) = carry

# Online softmax update per iteration:
logit_t = w_t · RMSNorm(h_t)              # Per-block learned query
new_max = max(lse, logit_t)
acc = exp(lse - new_max) · acc + exp(logit_t - new_max) · h_t
lse = new_max + log(exp(lse - new_max) + exp(logit_t - new_max))

# After loop: normalize
x_coda = acc / exp(lse)                    # Depth-weighted combination
```

- Full gradient flow through `jax.checkpoint` (recomputed on backward)
- O(1) memory per entry (no stacking, no graph explosion)
- Pallas kernel available for TPU-optimized forward+backward
- **Disabled in PyTorch preflight** (can't efficiently checkpoint scan carry)

---

## Training Pipeline (70k Steps)

```
Phase         Steps       What Happens
─────────────────────────────────────────────────────────
DENSE         0 → 10k     Full-stack training at density=1.0
                           CMS gradient scores accumulate
                           Neural memory warmup: off 0-1k, ramp 1k-5k

PRUNE         10k → 50k   Gradual tile pruning every 3k steps
                           10% of remaining tiles killed per round
                           13 rounds → 25.4% final density
                           Dead tiles re-zeroed after optimizer step

COMPACT       step 50k    Block-ELL rebuild: [R, K_new, 16, 16]
                           Optimizer reset (fresh momentum/variance)
                           torch.compile / XLA recompile

SETTLE        50k → 52k   Train compacted model, let loss stabilize

ROUTE         52k → 70k   ReMoE routing over 16 tile-group clusters
                           L1 regularization drives 50% cluster sparsity
                           Iteration embeddings condition the router
```

---

## Dimensions (Preflight Config)

```
d_model       = 512
d_ff          = 1376       (8/3 × 512, rounded to 16)
n_heads       = 8          (query heads)
n_kv_heads    = 2          (KV heads, GQA 4:1)
head_dim      = 64
vocab_size    = 49,152     (StarCoder2 tokenizer)
max_seq_len   = 1,024
n_prelude     = 3
n_core        = 6          (shared weights, looped T times)
n_coda        = 3
mean_depth    = 6          (T ~ Poisson(6), max 8)
tile_size     = 16         (Block-ELL pruning granularity)
n_clusters    = 16         (ReMoE routing groups)
d_memory      = 1,024      (neural memory MLP hidden dim)
n_mem_layers  = 4          (neural memory MLP depth)

Total params  ≈ 66.6M     (dense, pre-pruning)
Effective depth = 3 + 6×6 + 3 = 42 layers (at mean T=6)
```
