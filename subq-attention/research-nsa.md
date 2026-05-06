# DeepSeek NSA: Native Sparse Attention — Reverse Engineering Notes

**Paper**: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
**ArXiv**: https://arxiv.org/abs/2502.11089 (Feb 2025)
**ACL 2025 Best Paper**
**Authors**: DeepSeek-AI + Peking University (Liang Wenfeng, Yang Yaodong et al.)

---

## 1. NSA Three-Branch Architecture

NSA processes each query through **three parallel attention branches** whose outputs
are combined via learned gating:

```
o_t = SUM_{c in {cmp, slc, win}} g_t^c * Attn(q_t, K_t^c, V_t^c)
```

where g_t^c are per-token gate scores (described in Section 2).

### 1.1 Compressed Branch (cmp)

**Purpose**: Coarse-grained global context. Captures broad semantic patterns.

**Mechanism**: Groups consecutive KV tokens into compressed block representations
using a **learned MLP** with intra-block position encoding.

**Formula**:
```
K_cmp_t = { phi(k_{i*d+1 : i*d+l}) | 1 <= i <= floor((t-l)/d) }
```

where `phi` is a learnable MLP-based aggregator. NOT simple mean-pooling -- it has
learned weights with positional encoding within each block.

**Hyperparameters** (from the 27B model):
- Block length: **l = 32** (each compressed token represents 32 raw tokens)
- Sliding stride: **d = 16** (50% overlap between blocks)
- Effective compression ratio: ~16x (stride controls output count)
- Output per compressed token: R^{d_k} (same dim as original keys)

**What phi actually is**: An MLP that takes l=32 token embeddings as input and
produces one compressed key and one compressed value. Intra-block position encoding
means each position within the block gets a learned bias before aggregation, so the
MLP can attend to position-specific features (e.g., first token of block vs last).

### 1.2 Selected Branch (slc) -- THE QUADRATIC BOTTLENECK

**Purpose**: Fine-grained retrieval of the most important token blocks.

**Mechanism**: Selects top-n most important blocks using importance scores derived
from the **already-computed compressed attention**.

**Three-step selection process**:

**Step 1 -- Compute block importance scores from compressed attention**:
```
p_cmp_t = softmax(q_t^T * K_cmp_t)     # <-- THIS IS O(T/d) per query, O(T^2/d) total
```
This is the compressed branch's attention weights. Each weight tells how important
that compressed block is for this query.

**Step 2 -- Aggregate to selection-block granularity**:
When compression and selection blocks align (l' = l = d), the scores transfer directly:
```
p_slc_t = p_cmp_t
```
For GQA/MQA, importance is shared across heads in a group:
```
p_slc'_t = SUM_{h=1}^{H} p_slc_t^{(h)}    # sum across heads in GQA group
```

**Step 3 -- Top-n block selection + KV gather**:
```
I_t = { i | rank(p_slc'_t[i]) <= n }       # top-n most important blocks
K_slc_t = Cat[ k_{i*l'+1 : (i+1)*l'} for i in I_t ]   # gather FULL-RESOLUTION keys
V_slc_t = Cat[ v_{i*l'+1 : (i+1)*l'} for i in I_t ]   # gather FULL-RESOLUTION values
```

**Hyperparameters**:
- Selection block size: **l' = 64** tokens
- **n = 16** blocks selected (includes 1 fixed initial block + 2 local blocks = 13 free choices)
- Total selected tokens per query: n * l' = 16 * 64 = **1024 tokens**

**Critical insight**: The selected branch does NOT recompute Q*K scores for the full
sequence. It **reuses** the compressed branch's attention scores as a proxy for
block importance. This is the key efficiency trick -- compute full attention only
over the 1024 selected tokens.

### 1.3 Sliding Window Branch (win)

**Purpose**: Preserve local context, prevent "shortcut learning" where the model
over-relies on compressed/selected representations.

**Formula**:
```
K_win_t = k_{t-w : t}
V_win_t = v_{t-w : t}
```

**Hyperparameters**:
- Window size: **w = 512** tokens
- Standard causal masking within window

---

## 2. Gating Mechanism

The three branch outputs are combined via **learned per-token gating**:

```
g_t = sigmoid(MLP(x_t))      # x_t is the input features (likely query or pre-attention hidden state)
# g_t is a 3-element vector: [g_cmp, g_slc, g_win]

o_t = g_cmp * Attn(q_t, K_cmp, V_cmp) + g_slc * Attn(q_t, K_slc, V_slc) + g_win * Attn(q_t, K_win, V_win)
```

**Properties**:
- Sigmoid activation (NOT softmax) -- gates are independent, can all be high or all low
- Per-token: each token position gets its own gate values
- Per-head is not explicitly stated; likely per-token across all heads
- The MLP architecture is not detailed in the paper (likely small: Linear -> GELU -> Linear -> 3)
- "Stable learning by preventing gradient interference between local and long-range pattern recognition"

---

## 3. The Quadratic Bottleneck -- Precisely Located

**WHERE is the O(n^2) computation?**

There are TWO places where computation scales with full sequence length:

### 3a. Compressed branch attention: O(T^2 / d)

```
p_cmp_t = softmax(q_t^T * K_cmp_t)    # for each query q_t, dot with ALL compressed keys
```

For T queries, each attending to T/d compressed keys:
- Per query: O(T/d * d_k) = O(T * d_k / d)
- All queries: O(T^2 * d_k / d)

With d=16 and T tokens, this is O(T^2/16) -- still quadratic, just 16x cheaper.
For T=64k, this is 64k * 4k = 256M dot products per head.

### 3b. Selected branch attention: O(T * n * l')

This is the CHEAP part: each query attends to only n*l' = 1024 tokens.
- Per query: O(1024 * d_k)
- All queries: O(T * 1024 * d_k) = O(T) -- linear!

### 3c. Sliding window: O(T * w)

- Per query: O(512 * d_k)
- Linear in T.

**CONCLUSION**: The quadratic bottleneck is in **Step 1 of the selection process** --
computing the compressed attention scores `p_cmp_t = softmax(q_t^T * K_cmp_t)`.
Although this is reduced by the compression ratio d=16, it remains **O(T^2/d) = O(T^2)**
asymptotically.

**This is what we want to replace.** The selected branch's actual attention computation
is already linear. The bottleneck is the SCORING step that decides WHICH blocks to select.

---

## 4. Training Details

**Model**: 27B total parameters, 3B active (MoE architecture)

**Pretraining**:
- 270B tokens on 8k-length texts
- Extended to 32k with YaRN RoPE scaling
- Trained to full convergence
- Standard cross-entropy loss (no special modifications)

**Key property -- Natively Trainable**:
- All operations are differentiable (softmax scores, top-n selection via straight-through or differentiable sorting)
- Trained from scratch with NSA -- NOT post-hoc sparsification of a dense model
- Loss curves show NSA consistently outperforms Full Attention (not just matches)
- No two-stage warm-up needed (unlike DSA which needs dense warm-up)

**GQA Configuration** (from paper's 27B model):
| Parameter | Value |
|-----------|-------|
| GQA groups | 4 |
| Heads per group | 16 |
| Total heads | 64 |
| d_q, d_k | 192 |
| d_v | 128 |

**Speed Results** (at 64k tokens):
| Operation | Speedup vs Dense |
|-----------|-----------------|
| Forward | 9.0x |
| Backward | 6.0x |
| Decoding | 11.6x |

---

## 5. NSA -> V3.2 DSA -> V4 CSA+HCA Evolution

### 5a. V3.2: DeepSeek Sparse Attention (DSA) -- September 2025

DSA replaced NSA's three-branch design with a **two-stage** approach:

**Lightning Indexer** (replaces NSA's compressed-branch scoring):
- Separate set of small attention heads: H_I = 64 indexer heads, d_I = 128 dim
- Queries and keys projected into indexer space from the compressed query activation c^Q
- **Hadamard rotation** applied to decorrelate features for low-precision robustness
- **FP8 quantization** (NOT FP4 -- FP4 came in V4):
  ```
  (q_fp8, s_q) = quant8(Hadamard(q_idx))
  (k_fp8, s_k) = quant8(Hadamard(k_idx))
  ```
- **ReLU scoring** (not softmax!):
  ```
  logits_{h,t} = q_fp8_h * k_fp8_t
  logits+_{h,t} = max(0, logits_{h,t})       # ReLU discards negative correlations
  score_t = SUM_h w_h * logits+_{h,t}        # w_h = learned scalar gates
  index_score_t = score_t * s_k(t)           # dequant scale restores magnitude
  ```
- Top-k = **2048 tokens** selected per query
- Block size = **64** (mandatory)

**Key difference from NSA**: The Lightning Indexer is a **separate cheap neural network**
that runs in FP8 with tiny heads. NSA reused compressed attention scores (which were
part of an actual attention branch). DSA decouples scoring from attention entirely.

**Complexity**:
- Indexer: O(T) scan in small FP8 space -- still linear in T per query, O(T^2) total,
  but with ~100x smaller constant (FP8, 128-dim, 64 heads vs full precision, 192-dim)
- Main attention: O(T * k) = O(T * 2048) -- linear

**Training**: Required **dense warm-up** followed by sparse training (unlike NSA which
trained from scratch). Hint that NSA's native training was hard to stabilize at scale.

### 5b. V4: CSA + HCA -- April 2026

V4 split attention into two interleaved mechanisms:

#### Compressed Sparse Attention (CSA) -- evolved from DSA

1. **KV Compression**: Groups of 8 tokens -> 1 KV entry via **learned softmax-gated pooling**
   with stride=4. This gives **4x KV reduction** (vs NSA's 16x compression).
   More conservative compression = higher quality.

2. **Lightning Indexer**: Upgraded to **FP4** (from DSA's FP8). Multi-head ReLU-scored
   dot product. Selects **top-k = 512** (V4-Pro) or top-1024 compressed entries per query.
   Since entries represent 8 tokens each, this covers 4096-8192 raw tokens.

3. **Sliding window**: Last **128 uncompressed tokens** attended directly (vs NSA's 512).

4. **Shared KV vectors** across heads (like MQA) + inverse RoPE correction.

5. **Attention sinks**: Learnable sink logits added to softmax denominator.
   Attention scores can sum to < 1, expressing "nothing relevant here."

#### Heavily Compressed Attention (HCA) -- NEW in V4

1. **KV Compression**: **128 tokens -> 1 entry**, stride=128. **128x KV reduction.**
   At 1M tokens, this yields ~7,800 compressed entries.

2. **Dense attention** over the compressed sequence -- no sparse selection needed.
   7,800 entries is small enough for O(T * 7800) = O(T) effective cost.

3. **Dense MQA** (Multi-Query Attention) over compressed blocks.

4. **Sliding window**: Same 128-token window as CSA.

#### Layer Interleaving

**V4-Pro** (61 layers):
```
Layers 0-1:    HCA only (broad global context first)
Layers 2-60:   Alternating CSA <-> HCA
MTP block:     Sliding-window only
```

**V4-Flash** (43 layers):
```
Layers 0-1:    Sliding window
Layers 2-42:   Alternating CSA <-> HCA
```

#### Why V4 is Subquadratic (and NSA is Not)

| Step | NSA | V4 CSA |
|------|-----|--------|
| Importance scoring | softmax(q^T * K_cmp) over all T/d compressed tokens = **O(T^2/d)** | Lightning Indexer in FP4 over T/4 compressed tokens = **O(T^2/4)** but ~1000x cheaper constant |
| Fine attention | O(T * n*l') = O(T * 1024) | O(T * k) = O(T * 512*8) = O(T * 4096) |
| Window attention | O(T * 512) | O(T * 128) |

**Honest assessment**: V4 CSA is still technically O(T^2) in the indexer, but with such
a small constant (FP4, tiny heads, ReLU not softmax) that it behaves subquadratically
in practice. The REAL subquadratic win comes from **HCA**, which compresses 128x and
then does dense attention over only T/128 entries = genuinely O(T * T/128) = O(T^2/128).

At 1M tokens: T^2/128 = ~7.8B ops for HCA (vs T^2 = 10^12 for dense). Combined with
the Lightning Indexer's tiny constant, the effective complexity is manageable.

**The "subquadratic" claim is about practical scaling, not asymptotic complexity.**

---

## 6. Pseudocode -- Full NSA Forward Pass

```python
def nsa_attention_forward(x, W_q, W_k, W_v, W_o, gate_mlp, compressor_mlp):
    """
    NSA forward pass for a single layer.
    
    x: [B, T, D] input hidden states
    Returns: [B, T, D] output
    """
    B, T, D = x.shape
    
    # ─── Project Q, K, V ─────────────────────────────────────────────────
    q = W_q(x)  # [B, T, H, d_k]
    k = W_k(x)  # [B, T, H_kv, d_k]   (H_kv = H/group_size for GQA)
    v = W_v(x)  # [B, T, H_kv, d_v]
    q, k = apply_rope(q, k)
    
    # ─── Hyperparameters ─────────────────────────────────────────────────
    l = 32        # compression block length
    d = 16        # compression stride (50% overlap)
    l_prime = 64  # selection block size
    n = 16        # number of blocks to select
    w = 512       # sliding window size
    
    # ─── Branch 1: Compressed Attention ──────────────────────────────────
    # Compress KV into block representations
    n_blocks = (T - l) // d
    K_cmp = []  # will have n_blocks entries
    V_cmp = []
    for i in range(n_blocks):
        start = i * d
        end = start + l
        # phi = learned MLP with intra-block positional encoding
        k_block = k[:, start:end, :, :]          # [B, l, H_kv, d_k]
        v_block = v[:, start:end, :, :]          # [B, l, H_kv, d_v]
        K_cmp.append(compressor_mlp(k_block))    # -> [B, 1, H_kv, d_k]
        V_cmp.append(compressor_mlp(v_block))    # -> [B, 1, H_kv, d_v]
    K_cmp = stack(K_cmp, dim=1)  # [B, n_blocks, H_kv, d_k]
    V_cmp = stack(V_cmp, dim=1)  # [B, n_blocks, H_kv, d_v]
    
    # Compressed attention (with causal masking)
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  THIS IS THE O(T^2/d) QUADRATIC BOTTLENECK                     ║
    # ║  Every query attends to ALL compressed keys.                    ║
    # ║  T queries x T/d keys = T^2/d dot products per head.           ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    attn_cmp = softmax(q @ K_cmp.T / sqrt(d_k))  # [B, H, T, n_blocks]
    out_cmp = attn_cmp @ V_cmp                     # [B, H, T, d_v]
    
    # ─── Branch 2: Selected Attention ────────────────────────────────────
    # Reuse compressed attention scores as block importance
    # (This is the clever trick -- no extra Q*K computation needed)
    p_cmp = attn_cmp  # [B, H, T, n_blocks]
    
    # For GQA: aggregate importance across heads in each group
    p_slc = p_cmp.sum(dim=1)  # [B, T, n_blocks] -- sum across heads
    
    # Select top-n blocks per query position
    # (1 initial block + 2 local blocks are always included)
    _, top_indices = p_slc.topk(n, dim=-1)  # [B, T, n]
    
    # Gather full-resolution KV for selected blocks
    K_slc = gather_blocks(k, top_indices, block_size=l_prime)  # [B, T, n*l', H_kv, d_k]
    V_slc = gather_blocks(v, top_indices, block_size=l_prime)  # [B, T, n*l', H_kv, d_v]
    
    # Fine-grained attention over selected tokens only -- O(T * n * l') = O(T)
    out_slc = attention(q, K_slc, V_slc, causal=True)  # [B, H, T, d_v]
    
    # ─── Branch 3: Sliding Window ────────────────────────────────────────
    # Each query attends to previous w tokens -- O(T * w)
    out_win = sliding_window_attention(q, k, v, window_size=w)  # [B, H, T, d_v]
    
    # ─── Gating & Combination ────────────────────────────────────────────
    # Learned per-token gates via sigmoid MLP
    gates = sigmoid(gate_mlp(x))  # [B, T, 3]
    g_cmp = gates[:, :, 0:1]     # [B, T, 1]
    g_slc = gates[:, :, 1:2]
    g_win = gates[:, :, 2:3]
    
    out = g_cmp * out_cmp + g_slc * out_slc + g_win * out_win  # [B, H, T, d_v]
    
    # ─── Output Projection ──────────────────────────────────────────────
    out = reshape(out, [B, T, H * d_v])
    return W_o(out)
```

---

## 7. Our Replacement Targets

To replace the quadratic bottleneck (Section 3a), we need a mechanism that can
score ALL compressed blocks for each query WITHOUT computing the full dot product.

**Candidate subquadratic selectors** (from our ablation plan):

| Method | How it scores | Complexity | Notes |
|--------|--------------|------------|-------|
| LSH | Hash q and K_cmp into buckets | O(T * d_k) | Noisy, needs multiple rounds |
| Product Keys (PEER) | Factor keys into subspaces, lookup per subspace | O(T * sqrt(K)) | Clean, differentiable |
| Learned Block Router | Small MLP scores blocks directly | O(T * n_blocks * d_router) | Our existing approach |
| Lightning Indexer (DSA) | FP8 tiny-head dot product + ReLU | O(T^2/d) but ~1000x cheaper | Still quadratic, just fast |
| Hierarchical (LoD) | Multi-resolution tree search | O(T * log T) | Complex, novel |

**The Lightning Indexer (DSA/V4 approach)** is NOT truly subquadratic -- it's "practically
subquadratic" by making the quadratic step so cheap it doesn't matter. This is a valid
engineering approach but doesn't solve the asymptotic problem.

**Our goal**: Find a TRULY subquadratic selector that maintains NSA's selection quality.
The PEER/product-key approach is the most promising: factor the key space into two
halves, do O(sqrt(K)) lookup per subspace, combine with a Cartesian product to recover
top-k. This is O(T * sqrt(n_blocks)) per query = O(T * sqrt(T/d)) = O(T^{3/2} / sqrt(d)).

---

## 8. Summary Table

| Component | NSA (Feb 2025) | DSA / V3.2 (Sep 2025) | V4 CSA+HCA (Apr 2026) |
|-----------|---------------|----------------------|----------------------|
| Branches | 3 (cmp + slc + win) | 2 (indexer + main attn + window) | 2 interleaved types (CSA + HCA) |
| Compression | MLP, l=32, d=16 | Inherited from MLA | Softmax-gated pooling, 4x (CSA) / 128x (HCA) |
| Selection | Reuse cmp scores, top-16 blocks | Lightning Indexer FP8, top-2048 tokens | Lightning Indexer FP4, top-512 cmp entries |
| Window | 512 tokens | Unspecified | 128 tokens |
| Gating | Sigmoid MLP, per-token | N/A (single path) | Per-layer type assignment |
| Training | Native from scratch | Dense warm-up -> sparse | From scratch (32T+ tokens) |
| Quadratic? | Yes: O(T^2/16) | Yes but ~1000x cheaper constant | Yes but FP4 + HCA amortizes |
| Sinks | No | No | Learnable attention sinks |

---

## Sources

- [NSA Paper (arXiv)](https://arxiv.org/abs/2502.11089)
- [NSA Paper (ACL Anthology)](https://aclanthology.org/2025.acl-long.1126/)
- [NSA Paper HTML](https://arxiv.org/html/2502.11089v1)
- [DeepSeek V4 Technical Report](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)
- [DeepSeek V4 HuggingFace Blog](https://huggingface.co/blog/deepseekv4)
- [DeepSeek V4-Pro Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DSA Technical Blog (Leon Ericsson)](https://leonericsson.github.io/blog/2025-10-16-dsa)
- [NSA to DSA Evolution (Champaign Magazine)](https://champaignmagazine.com/2025/09/30/ai-on-ai-sparse-attention-from-nsa-to-dsa/)
- [V4 CSA+HCA (MarkTechPost)](https://www.marktechpost.com/2026/04/24/deepseek-ai-releases-deepseek-v4-compressed-sparse-attention-and-heavily-compressed-attention-enable-one-million-token-contexts/)
- [Decoding DeepSeek-V4 (OutcomeSchool)](https://outcomeschool.com/blog/decoding-deepseek-v4)
- [V4 Hybrid Attention (dasroot.net)](https://dasroot.net/posts/2026/04/deepseek-v4-hybrid-attention-massive-contexts/)
- [vLLM DSA Implementation](https://vllm.ai/blog/deepseek-v3-2)
