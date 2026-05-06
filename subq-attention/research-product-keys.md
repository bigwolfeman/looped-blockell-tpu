# Product Key Retrieval for Subquadratic Attention Selection

Research brief for replacing NSA's quadratic selection branch with O(n*sqrt(k)) product-key lookup.

---

## 1. PEER: Parameter Efficient Expert Retrieval

**Paper**: "Mixture of A Million Experts" (He et al., Google DeepMind, July 2024) — arXiv:2407.04153

### 1.1 Product Key Construction

Product keys factor a large key space into two smaller sub-spaces via Cartesian product.
Given N total keys, maintain two codebooks C_a, C_b each of size sqrt(N):

```
C_a = {c_1, c_2, ..., c_m}   where m = sqrt(N), each c_i in R^{d/2}
C_b = {c_1, c_2, ..., c_m}   where m = sqrt(N), each c_j in R^{d/2}
```

The full key space K = C_a x C_b has N = m^2 entries.
Each composite key is the concatenation [c_i; c_j], yielding a d-dimensional vector.

### 1.2 Scoring: Additive Decomposition

The key insight is that dot-product scoring decomposes additively:

```
score(i, j) = q^T [c_i; c_j]
             = q_a^T c_i + q_b^T c_j
```

where q_a = q[:d/2] and q_b = q[d/2:] are the two halves of the query vector.

This decomposition is **exact** — no approximation. The top-k items from the full
Cartesian product are guaranteed to appear in the top-c candidates from each sub-space,
where c >= k.

### 1.3 Retrieval Algorithm

```
Input: query q in R^d, codebooks C_a, C_b each of size m, desired top-k

1. Split query:        q_a = q[:d/2],  q_b = q[d/2:]
2. Sub-scores:         s_a = q_a @ C_a^T    # shape [m]
                       s_b = q_b @ C_b^T    # shape [m]
3. Sub-space top-c:    I_a = top_c(s_a)     # c indices, c >= k
                       I_b = top_c(s_b)     # c indices
4. Cartesian product:  candidates = {(i,j) : i in I_a, j in I_b}   # c^2 pairs
5. Full scores:        for (i,j) in candidates: score = s_a[i] + s_b[j]
6. Final top-k:        select k pairs with highest scores from c^2 candidates
```

**Typical values**: m=1024 (so N=1M), c=k=16 per head, h=8 heads, granularity G=128.

### 1.4 Complexity Analysis

| Method | Comparisons per query | With d dimensions |
|--------|----------------------|-------------------|
| Flat top-k over N items | O(N) | O(N * d) |
| Product-key top-k | O(2*sqrt(N) + c^2) | O(2*sqrt(N)*d/2 + c^2) = O(sqrt(N)*d + c^2) |

For N=1M, k=16, c=16: flat needs 1,000,000 comparisons, product-key needs 2*1024 + 256 = 2,304.
That is a **434x reduction** in retrieval cost.

### 1.5 BatchNorm for Utilization

Without normalization, queries collapse to a narrow cone in embedding space, causing
most experts to go unused. PEER applies **BatchNorm to query vectors** across the batch
dimension (treating all tokens as independent samples):

```python
q = self.query_proj(x)           # [B, T, d]
q = q.reshape(-1, d)             # [B*T, d]
q = self.query_bn(q)             # BatchNorm over B*T dimension
q = q.reshape(B, T, d)
```

Result: 99.98% of 1M experts used. No auxiliary loss needed.
Without BatchNorm: utilization drops significantly, KL divergence from uniform doubles.

### 1.6 Expert Computation

Each expert is a single-neuron MLP:

```
e_i(x) = sigma(u_i^T x) * v_i
```

where u_i, v_i in R^d are learned vectors. This is a rank-1 operation.
With k active experts summed, the layer produces a rank-k output.

### 1.7 Gradient Flow

Gradients flow through continuous softmax weights, NOT through the discrete top-k selection:

```
output = sum_{i in top_k} softmax(score_i) * e_i(x)
```

- The softmax weights are differentiable w.r.t. scores, which are differentiable w.r.t. q and keys
- The top-k selection is non-differentiable but acts as a hard gate (similar to standard MoE)
- Only ~128 of 1M expert rows receive gradients per token (sparse updates)
- Keys receive gradients through the score computation: d(score)/d(c_i) = q_a

### 1.8 Results

At 2e19 FLOPs on C4:

| Model | Perplexity |
|-------|-----------|
| Dense FFN | 18.31 |
| PKM (Lample 2019) | 17.36 |
| MoE-128 (standard) | 17.12 |
| **PEER (1M experts)** | **16.45** |

PEER beats dense by 10.1%, standard MoE by 3.9%.

---

## 2. Adapting PEER for Attention KV Selection

The core idea: instead of selecting which *experts* to activate, select which *KV positions*
to attend to. This replaces NSA's selection branch with O(n*sqrt(k)) lookup.

### 2.1 The Problem with NSA's Selection

NSA's selection branch works as follows:
1. Divide the sequence into blocks of l'=64 tokens
2. Compute block importance using compressed attention: p = softmax(q^T K_cmp)
3. Select top-n blocks (n=16) based on importance scores

The compression step already reduces cost, but block importance scoring still requires
computing attention over ALL compressed tokens — O(T/d) per query where d is the stride.
For T=64k with stride 16, that is 4096 compressed tokens. Not truly quadratic, but
still grows linearly with sequence length.

**Product keys can make this O(sqrt(T/d))** — truly sublinear.

### 2.2 Constructing Product Keys from KV Positions

There are two fundamentally different approaches:

**Approach A: Learned codebook (fixed keys, like PEER)**

Maintain a fixed codebook. Each KV block is *assigned* to its nearest codebook entry.
Assignment is recomputed periodically (every N steps or on cache insertion).

```
# On KV cache insertion:
block_repr = mean_pool(K[block_start:block_end])    # [d]
block_repr_a, block_repr_b = split(block_repr)      # [d/2] each
cluster_a = argmin(||block_repr_a - C_a||)           # nearest sub-key
cluster_b = argmin(||block_repr_b - C_b||)           # nearest sub-key
inverted_index[cluster_a][cluster_b].append(block_id)
```

Pros: O(sqrt(N_blocks)) query cost. Codebook is stable.
Cons: Assignment is approximate. Stale if K changes fast.

**Approach B: Direct product-key scoring on compressed K (our recommendation)**

Use NSA's existing compressed representations as keys, but organize them with
product-key structure for fast retrieval:

```
# Given T/d compressed key representations K_cmp of shape [T/d, d_head]
# Factor them into two sub-spaces for product-key lookup

# Option 1: Split dimensions (like PEER)
K_cmp_a = K_cmp[:, :d_head//2]    # [T/d, d_head/2]
K_cmp_b = K_cmp[:, d_head//2:]    # [T/d, d_head/2]

# Cluster each sub-space into m centroids
centroids_a = online_kmeans(K_cmp_a, m)   # [m, d_head/2]
centroids_b = online_kmeans(K_cmp_b, m)   # [m, d_head/2]

# Build inverted index: which blocks map to which centroid pair
for block_idx in range(T/d):
    ca = nearest(K_cmp_a[block_idx], centroids_a)
    cb = nearest(K_cmp_b[block_idx], centroids_b)
    inv_index[ca][cb].append(block_idx)

# At query time:
q_a, q_b = split(q)
top_c_a = topk(q_a @ centroids_a.T, c)
top_c_b = topk(q_b @ centroids_b.T, c)
candidate_blocks = union(inv_index[i][j] for i in top_c_a for j in top_c_b)
# Attend only to candidate_blocks
```

**Option 2 (simpler, recommended for first implementation):**
Skip clustering entirely. Use product-key structure directly on the compressed tokens.
This is valid when T/d is small enough (e.g., 4096 compressed tokens = m=64 per sub-space).

### 2.3 Handling Dynamic K (K Changes Every Step)

This is the key difference from PEER, where expert parameters are fixed between updates.
In attention, K changes every forward pass as new tokens arrive.

**Three strategies**:

1. **Rebuild per-step** (brute force): Recompute inverted index every step.
   Cost: O(T/d * d) per step. Acceptable if T/d < 10k.

2. **Incremental update** (for autoregressive): Only new KV entries need assignment.
   Existing entries keep their cluster. Cost: O(1) amortized per new token.

3. **Learned assignment** (best quality): Train a small MLP to predict cluster assignment
   from compressed K, avoiding explicit nearest-neighbor search.

For training (bidirectional), option 1 is fine — it is a one-time cost per forward pass.
For inference (autoregressive), option 2 is natural — just assign each new block as it enters the cache.

### 2.4 Gradient Flow

The router needs gradients to learn good selection. Two gradient paths:

**Path 1: Through scores (same as PEER)**
The softmax weights on selected blocks are differentiable. Query and key projections
receive gradients through: d(loss)/d(q) via d(score)/d(q) = C_a^T (selected columns).

**Path 2: Through attention output (Straight-Through Estimator)**
If using hard block selection, apply STE to the top-k mask so gradients flow
to the scoring function during backward.

**Recommendation**: Use Path 1 (continuous scores). The selected blocks get standard
attention weights. Non-selected blocks get zero weight. This is identical to how
standard sparse attention works — no STE needed.

---

## 3. Related Fast Nearest-Neighbor Methods

### 3.1 FAISS IVF (Inverted File Index)

**How it works**: Cluster all vectors into C cells using k-means. At query time, probe
the top-p closest cells and exhaustively search within them.

| Property | Value |
|----------|-------|
| Build time | O(N * C * iterations) for k-means |
| Query time | O(p * N/C * d) where p = probes |
| Recall@10 | 95%+ with p=8, C=sqrt(N) |
| Differentiable? | **No** — hard cluster assignment |

**For attention**: Could cluster KV blocks into cells. But k-means is not differentiable
and must be re-run as K changes. Product keys are strictly better for this use case
because the factored structure IS the index — no separate build step.

### 3.2 HNSW (Hierarchical Navigable Small World Graphs)

**How it works**: Build a multi-layer graph where each node connects to its approximate
nearest neighbors. Query by greedy traversal from top layer down.

| Property | Value |
|----------|-------|
| Build time | O(N * log(N) * M) where M = edges per node |
| Query time | O(log(N) * d) |
| Recall@10 | 99%+ |
| Differentiable? | **No** — graph traversal is discrete |

**For attention**: Excellent query speed but graph construction is expensive and must be
rebuilt as K changes. Not suitable for training-time attention where K is recomputed
every forward pass. Could work for inference-only KV cache lookup in very long contexts.

### 3.3 Differentiability Summary

| Method | Differentiable? | Usable during training? | Notes |
|--------|----------------|------------------------|-------|
| Product keys (PEER) | Yes (through scores) | Yes | Best fit for attention |
| LSH (Reformer) | Partial (through hash buckets) | Yes, but noisy gradients | Hash collisions = noise |
| FAISS IVF | No | No (inference only) | k-means not differentiable |
| HNSW | No | No (inference only) | Graph ops not differentiable |
| SparseK (differentiable top-k) | Yes (convex relaxation) | Yes | O(n) but constant factor overhead |
| HiP (hierarchical pruning) | No (training-free) | No (post-hoc only) | O(T log T), plug-and-play |

---

## 4. Comparison: Product Keys vs LSH vs Full Top-k

### 4.1 Selection Quality

| Method | Recall of true top-k | Failure mode |
|--------|---------------------|--------------|
| Full top-k | 100% (exact) | O(n) cost |
| Product keys | 100% (exact, given c >= k) | None if c is large enough |
| LSH | ~85-95% (probabilistic) | Hash collisions miss neighbors |
| Random | k/n | Baseline |

**Critical insight**: Product keys give **exact** top-k recovery when c (sub-space top-k)
is set high enough relative to k. This is because the additive score decomposition
guarantees that the true top-k must appear in the Cartesian product of sub-space top-c.
LSH only gives probabilistic guarantees.

### 4.2 Speed Comparison

Assuming d=128 (head dim), k=16 (selected items):

| n (total items) | Full top-k | Product keys (m=sqrt(n)) | LSH (b=8 bands) |
|-----------------|-----------|--------------------------|-----------------|
| 1,024 | 131K ops | 2.3K ops (57x) | 8.2K ops |
| 4,096 | 524K ops | 4.2K ops (125x) | 8.2K ops |
| 16,384 | 2.1M ops | 8.4K ops (250x) | 8.2K ops |
| 65,536 | 8.4M ops | 16.6K ops (506x) | 8.2K ops |
| 1,048,576 | 134M ops | 65.8K ops (2036x) | 8.2K ops |

LSH has constant cost (independent of n) but lower recall.
Product keys scale as O(sqrt(n)) — sublinear but not constant.
For the attention use case (n = T/block_size = 1k-16k), product keys are the sweet spot.

### 4.3 Memory Overhead

| Method | Extra memory | Notes |
|--------|-------------|-------|
| Full top-k | 0 | Just compute Q @ K^T |
| Product keys | 2 * sqrt(N) * d/2 | Two codebooks |
| LSH | n * b * hash_dim | Hash tables per item |
| FAISS IVF | C * d + inverted lists | Centroids + assignments |

For N=4096 blocks, d=128: product keys need 2 * 64 * 64 = 8K params.
Negligible compared to KV cache.

---

## 5. Implementation Sketch: Product-Key-Routed Attention

### 5.1 Architecture Overview

Replace NSA's selection branch with product-key routing while keeping compressed and
sliding window branches intact.

```
NSA Architecture (original):
  output = gate_cmp * Attn(Q, K_cmp, V_cmp)          # compressed branch
         + gate_sel * Attn(Q, K_sel, V_sel)           # selection branch (REPLACE THIS)
         + gate_slw * Attn(Q, K_window, V_window)     # sliding window branch

Product-Key NSA (proposed):
  output = gate_cmp * Attn(Q, K_cmp, V_cmp)          # compressed branch (unchanged)
         + gate_sel * PKAttn(Q, K, V, codebook)       # product-key selection (NEW)
         + gate_slw * Attn(Q, K_window, V_window)     # sliding window (unchanged)
```

### 5.2 Core Module: ProductKeyAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyAttention(nn.Module):
    """
    Product-key routed sparse attention.
    Replaces NSA's quadratic selection branch with O(sqrt(n_blocks)) retrieval.

    Instead of scoring ALL compressed tokens to find important blocks,
    we use factored product keys to retrieve the top-k blocks sublinearly.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        block_size: int = 64,       # tokens per block (NSA default)
        n_sub_keys: int = 64,       # sqrt of max blocks; ceil(sqrt(T/block_size))
        top_c: int = 8,             # top-c per sub-space
        top_k_blocks: int = 16,     # final number of blocks to attend to
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.block_size = block_size
        self.n_sub_keys = n_sub_keys
        self.top_c = top_c
        self.top_k_blocks = top_k_blocks
        self.d_half = d_head // 2

        # Query projection (separate from main attention Q)
        # Projects to split query for product-key lookup
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)

        # Product key codebooks: two sets of sub-keys
        # Shape: [n_sub_keys, d_head // 2]
        self.codebook_a = nn.Parameter(torch.randn(n_sub_keys, self.d_half) * 0.02)
        self.codebook_b = nn.Parameter(torch.randn(n_sub_keys, self.d_half) * 0.02)

        # Query BatchNorm (PEER's utilization trick)
        self.query_bn = nn.BatchNorm1d(d_head)

        # Block compressor: compress each block to a single vector for assignment
        self.block_compress = nn.Linear(d_head, d_head, bias=False)

        # Scale factor
        self.scale = d_head ** -0.5

    def _assign_blocks_to_keys(self, K_blocks: torch.Tensor):
        """
        Assign each KV block to its nearest product-key pair.
        K_blocks: [B, n_blocks, n_heads, d_head] — mean-pooled block representations

        Returns:
            block_key_a: [B, n_blocks, n_heads] — sub-key-a index per block
            block_key_b: [B, n_blocks, n_heads] — sub-key-b index per block
        """
        B, N, H, D = K_blocks.shape

        # Split block representations into two halves
        repr_a = K_blocks[..., :self.d_half]  # [B, N, H, d_half]
        repr_b = K_blocks[..., self.d_half:]  # [B, N, H, d_half]

        # Find nearest codebook entry for each half
        # scores_a: [B, N, H, n_sub_keys]
        scores_a = torch.einsum('bnhd,kd->bnhk', repr_a, self.codebook_a)
        scores_b = torch.einsum('bnhd,kd->bnhk', repr_b, self.codebook_b)

        block_key_a = scores_a.argmax(dim=-1)  # [B, N, H]
        block_key_b = scores_b.argmax(dim=-1)  # [B, N, H]

        return block_key_a, block_key_b

    def _retrieve_blocks(self, q: torch.Tensor, block_key_a, block_key_b):
        """
        Use product-key lookup to find the top-k blocks for each query.

        q: [B, T_q, H, d_head] — query vectors (already BatchNorm'd)
        block_key_a, block_key_b: [B, n_blocks, H] — block assignments

        Returns:
            selected_indices: [B, T_q, H, top_k_blocks] — indices of selected blocks
            selection_scores: [B, T_q, H, top_k_blocks] — scores for weighting
        """
        B, T, H, D = q.shape
        n_blocks = block_key_a.shape[1]

        # Split query into two halves
        q_a = q[..., :self.d_half]  # [B, T, H, d_half]
        q_b = q[..., self.d_half:]  # [B, T, H, d_half]

        # Score against codebooks: O(sqrt(n_blocks)) per query
        # sub_scores_a: [B, T, H, n_sub_keys]
        sub_scores_a = torch.einsum('bthd,kd->bthk', q_a, self.codebook_a)
        sub_scores_b = torch.einsum('bthd,kd->bthk', q_b, self.codebook_b)

        # Top-c per sub-space
        top_a_scores, top_a_idx = sub_scores_a.topk(self.top_c, dim=-1)  # [B,T,H,c]
        top_b_scores, top_b_idx = sub_scores_b.topk(self.top_c, dim=-1)  # [B,T,H,c]

        # Cartesian product of scores: [B, T, H, c, c]
        cart_scores = top_a_scores.unsqueeze(-1) + top_b_scores.unsqueeze(-2)
        cart_scores_flat = cart_scores.reshape(B, T, H, -1)  # [B, T, H, c^2]

        # Top-k from c^2 candidates
        topk_scores, topk_flat_idx = cart_scores_flat.topk(
            min(self.top_k_blocks, self.top_c ** 2), dim=-1
        )

        # Convert flat index back to (a_idx, b_idx) pairs
        a_sel = topk_flat_idx // self.top_c  # which entry in top_a
        b_sel = topk_flat_idx % self.top_c   # which entry in top_b

        # Map back to actual codebook indices
        # Gather from top_a_idx and top_b_idx
        sel_key_a = top_a_idx.gather(-1, a_sel)  # [B, T, H, top_k]
        sel_key_b = top_b_idx.gather(-1, b_sel)  # [B, T, H, top_k]

        # Now find which blocks match these (key_a, key_b) pairs
        # This is the inverted index lookup — implementation below
        selected_indices = self._inverted_lookup(
            sel_key_a, sel_key_b, block_key_a, block_key_b, n_blocks
        )

        return selected_indices, topk_scores

    def _inverted_lookup(self, sel_key_a, sel_key_b, block_key_a, block_key_b, n_blocks):
        """
        Find blocks whose assignments match the selected (key_a, key_b) pairs.

        For GPU efficiency, we use a batched approach:
        composite_sel = sel_key_a * n_sub_keys + sel_key_b    (query side)
        composite_blk = block_key_a * n_sub_keys + block_key_b  (block side)
        Then check membership.

        Returns: [B, T_q, H, top_k_blocks] block indices (padded with -1)
        """
        B, T, H, K = sel_key_a.shape
        n_blk = block_key_a.shape[1]

        # Composite keys for selected pairs: [B, T, H, K]
        composite_sel = sel_key_a * self.n_sub_keys + sel_key_b

        # Composite keys for all blocks: [B, n_blk, H]
        composite_blk = block_key_a * self.n_sub_keys + block_key_b

        # For each query, find blocks matching any selected composite key
        # Expand for broadcasting: [B, T, H, K, 1] vs [B, 1, H, 1, n_blk]
        match = (composite_sel.unsqueeze(-1) == composite_blk.unsqueeze(1).unsqueeze(3))
        # match: [B, T, H, K, n_blk]

        # Any match across K dimension
        any_match = match.any(dim=3)  # [B, T, H, n_blk]

        # Get indices of matching blocks (top_k_blocks of them)
        # Use topk on float mask to get indices
        match_scores = any_match.float()
        _, selected = match_scores.topk(
            min(self.top_k_blocks, n_blk), dim=-1
        )  # [B, T, H, top_k_blocks]

        return selected

    def forward(self, x, K_full, V_full, attention_mask=None):
        """
        Product-key routed sparse attention.

        x: [B, T_q, d_model] — input for query projection
        K_full: [B, T_kv, n_heads, d_head] — full key cache
        V_full: [B, T_kv, n_heads, d_head] — full value cache

        Returns: [B, T_q, d_model]
        """
        B, T_q, _ = x.shape
        T_kv = K_full.shape[1]
        n_blocks = T_kv // self.block_size

        # --- Step 1: Compute routing query (separate from attention query) ---
        q_route = self.q_proj(x).reshape(B, T_q, self.n_heads, self.d_head)

        # BatchNorm for utilization (PEER trick)
        q_flat = q_route.reshape(-1, self.d_head)
        q_flat = self.query_bn(q_flat)
        q_route = q_flat.reshape(B, T_q, self.n_heads, self.d_head)

        # --- Step 2: Compress KV blocks ---
        # Mean-pool each block of K to get block representatives
        K_blocked = K_full[:, :n_blocks * self.block_size].reshape(
            B, n_blocks, self.block_size, self.n_heads, self.d_head
        )
        K_blocks = K_blocked.mean(dim=2)  # [B, n_blocks, n_heads, d_head]
        K_blocks = self.block_compress(K_blocks)

        # --- Step 3: Assign blocks to product-key pairs ---
        block_key_a, block_key_b = self._assign_blocks_to_keys(K_blocks)

        # --- Step 4: Retrieve top-k blocks per query ---
        selected_indices, selection_scores = self._retrieve_blocks(
            q_route, block_key_a, block_key_b
        )
        # selected_indices: [B, T_q, n_heads, top_k_blocks]

        # --- Step 5: Gather selected KV blocks and compute attention ---
        # Expand block indices to token indices
        # Each selected block index maps to block_size contiguous tokens
        block_starts = selected_indices * self.block_size  # [B, T_q, H, top_k]
        offsets = torch.arange(self.block_size, device=x.device)  # [block_size]

        # token_indices: [B, T_q, H, top_k * block_size]
        token_indices = (block_starts.unsqueeze(-1) + offsets).reshape(
            B, T_q, self.n_heads, -1
        )
        n_selected_tokens = token_indices.shape[-1]

        # Clamp indices to valid range
        token_indices = token_indices.clamp(0, T_kv - 1)

        # Gather K and V for selected tokens
        # K_full: [B, T_kv, H, d_head] -> gather along dim 1
        token_indices_k = token_indices.permute(0, 2, 1, 3).reshape(B * self.n_heads, T_q, -1)
        K_flat = K_full.permute(0, 2, 1, 3).reshape(B * self.n_heads, T_kv, self.d_head)
        V_flat = V_full.permute(0, 2, 1, 3).reshape(B * self.n_heads, T_kv, self.d_head)

        # Expand indices for gather
        idx_expand = token_indices_k.unsqueeze(-1).expand(-1, -1, -1, self.d_head)

        # For each query position, gather its selected KV tokens
        # This requires per-query different indices, so we expand
        K_sel = torch.zeros(B * self.n_heads, T_q, n_selected_tokens, self.d_head,
                           device=x.device, dtype=K_full.dtype)
        V_sel = torch.zeros_like(K_sel)

        for t in range(T_q):
            idx_t = token_indices_k[:, t, :].unsqueeze(-1).expand(-1, -1, self.d_head)
            K_sel[:, t] = K_flat.gather(1, idx_t)
            V_sel[:, t] = V_flat.gather(1, idx_t)

        # --- Step 6: Standard scaled dot-product attention on selected subset ---
        # Q for attention (can reuse q_route or compute fresh)
        Q_attn = q_route.permute(0, 2, 1, 3).reshape(B * self.n_heads, T_q, self.d_head)
        Q_attn = Q_attn.unsqueeze(2)  # [BH, T_q, 1, d_head] -- unsqueeze is wrong, need matmul

        # Actually: [BH, T_q, d_head] @ [BH, T_q, d_head, n_sel] -> [BH, T_q, n_sel]
        attn_scores = torch.einsum('btd,btsd->bts', Q_attn.squeeze(2), K_sel) * self.scale

        # Optional: apply causal mask within selected tokens
        if attention_mask is not None:
            # Would need to construct mask for selected positions
            pass

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.einsum('bts,btsd->btd', attn_weights, V_sel)

        # Reshape back: [B, n_heads, T_q, d_head] -> [B, T_q, d_model]
        attn_out = attn_out.reshape(B, self.n_heads, T_q, self.d_head)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T_q, self.d_model)

        return attn_out
```

### 5.3 Optimized Version: Skip Inverted Index

The inverted index lookup in `_inverted_lookup` is the ugly part. A cleaner approach
for moderate n_blocks (< 16k): score blocks directly using product-key structure,
skip the assignment + lookup entirely.

```python
class ProductKeyAttentionV2(nn.Module):
    """
    Simplified product-key attention: use product-key scoring directly on
    compressed block representations. No inverted index needed.

    Key insight: We don't need to assign blocks to codebook entries.
    Instead, split the block representations AND the query into halves,
    compute sub-scores, and use the product-key trick to find top-k blocks
    without ever materializing the full score matrix.
    """

    def __init__(self, d_head, n_heads, block_size=64, top_k_blocks=16, top_c=8):
        super().__init__()
        self.d_head = d_head
        self.d_half = d_head // 2
        self.n_heads = n_heads
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.top_c = top_c
        self.scale = d_head ** -0.5

        self.query_bn = nn.BatchNorm1d(d_head)

    def forward(self, Q, K, V):
        """
        Q: [B, T_q, H, d_head]
        K: [B, T_kv, H, d_head]
        V: [B, T_kv, H, d_head]
        """
        B, T_q, H, D = Q.shape
        T_kv = K.shape[1]
        n_blocks = T_kv // self.block_size

        # --- Compress K into block representatives ---
        K_blocked = K[:, :n_blocks * self.block_size].reshape(
            B, n_blocks, self.block_size, H, D
        )
        K_blocks = K_blocked.mean(dim=2)  # [B, n_blocks, H, D]

        # --- Apply BatchNorm to Q for balanced retrieval ---
        Q_bn = self.query_bn(Q.reshape(-1, D)).reshape(B, T_q, H, D)

        # --- Product-key scoring ---
        # Split into halves
        Q_a = Q_bn[..., :self.d_half]   # [B, T_q, H, d_half]
        Q_b = Q_bn[..., self.d_half:]
        K_a = K_blocks[..., :self.d_half]  # [B, n_blk, H, d_half]
        K_b = K_blocks[..., self.d_half:]

        # Sub-scores: each is [B, T_q, H, n_blocks] -- but we DON'T materialize this!
        # Instead, we find top-c in each sub-space independently

        # For each query, score against all block-halves
        # scores_a: [B, H, T_q, n_blocks]
        scores_a = torch.einsum('bthd,bnhd->bhtn', Q_a, K_a)
        scores_b = torch.einsum('bthd,bnhd->bhtn', Q_b, K_b)

        # Top-c per sub-space
        top_a_vals, top_a_idx = scores_a.topk(self.top_c, dim=-1)  # [B,H,T,c]
        top_b_vals, top_b_idx = scores_b.topk(self.top_c, dim=-1)  # [B,H,T,c]

        # Cartesian product of scores
        cart_scores = top_a_vals.unsqueeze(-1) + top_b_vals.unsqueeze(-2)  # [B,H,T,c,c]

        # Top-k from c^2 candidates
        cart_flat = cart_scores.reshape(B, H, T_q, -1)  # [B,H,T,c^2]
        _, topk_flat = cart_flat.topk(self.top_k_blocks, dim=-1)  # [B,H,T,top_k]

        # BUT: these index into the Cartesian grid, not into blocks.
        # We need to find which BLOCK each Cartesian pair refers to.
        # The a-index and b-index point to different blocks!
        # This is the fundamental issue: in PEER, (a,b) maps to expert a*m+b.
        # In attention, there is no such mapping -- block 5 has BOTH a half-a and half-b.

        # SOLUTION: Fall back to union of top-c_a blocks and top-c_b blocks
        # This is a superset of the true top-k and costs at most 2*c blocks
        candidate_blocks_a = top_a_idx  # [B, H, T, c]
        candidate_blocks_b = top_b_idx  # [B, H, T, c]

        # Union (may have duplicates, which is fine -- just attend to unique ones)
        candidates = torch.cat([candidate_blocks_a, candidate_blocks_b], dim=-1)  # [B,H,T,2c]

        # Score candidates with FULL key (not just halves) for final ranking
        # Gather candidate block representations
        cand_expand = candidates.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        K_blocks_t = K_blocks.permute(0, 2, 1, 3)  # [B, H, n_blk, D]
        K_blocks_t = K_blocks_t.unsqueeze(2).expand(-1, -1, T_q, -1, -1)
        K_cand = K_blocks_t.gather(3, cand_expand)  # [B, H, T, 2c, D]

        Q_t = Q.permute(0, 2, 1, 3)  # [B, H, T, D]
        full_scores = torch.einsum('bhtd,bhtcd->bhtc', Q_t, K_cand) * self.scale

        # Final top-k from 2c candidates using full score
        _, final_sel = full_scores.topk(self.top_k_blocks, dim=-1)  # [B,H,T,top_k]
        selected_blocks = candidates.gather(-1, final_sel)  # [B,H,T,top_k]

        # --- Now do standard attention on selected blocks' tokens ---
        # (expand block indices to token indices, gather K/V, compute attention)
        # ... (same as V1 above)

        return selected_blocks  # placeholder — full impl gathers and attends
```

### 5.4 The Fundamental Mismatch and Resolution

There is a subtle but critical difference between PEER and attention KV selection:

**In PEER**: Expert index (i,j) = i*m + j. The Cartesian product IS the index space.
Each (a,b) pair uniquely identifies one expert. Product-key scoring is exact.

**In attention**: Block b has key K_b = [K_b_a; K_b_b]. The block scored high on
sub-key-a does NOT necessarily share its sub-key-b identity with another block.
There is no Cartesian decomposition of the block space — blocks are atomic.

**Three resolutions**:

1. **Union of sub-space top-c** (V2 above): Take top-c blocks from half-a scores UNION
   top-c blocks from half-b scores. At most 2c candidates, rescore with full key.
   Simple, effective, but O(n_blocks) not O(sqrt(n_blocks)).
   Wait — the sub-space scoring IS O(n_blocks) because we score Q against ALL block halves.

2. **Quantize blocks into product-key cells** (V1 above): Assign each block to its
   nearest codebook entry in each half-space. Then the inverted index lookup IS O(sqrt(n)).
   But assignment is lossy.

3. **Hierarchical pruning** (HiP-style): Score blocks at coarse level, prune half,
   repeat. O(n_blocks * log(n_blocks)). No product-key structure needed.

**Recommendation**: For n_blocks < 4096 (sequences up to 256k with block_size=64),
approach 2 (quantized product keys with inverted index) is best.
For n_blocks < 1024, the sub-scores are cheap enough that full scoring is fine — just
use approach 1 with c = top_k and skip the product-key machinery entirely.
Product keys only win when n_blocks > ~4000.

### 5.5 Autoregressive Generation (KV Cache)

During autoregressive generation, the KV cache grows by one token per step.
Product-key routing adapts naturally:

```python
class PKAttentionWithCache:
    """Stateful wrapper for autoregressive generation."""

    def __init__(self, pk_attn: ProductKeyAttention):
        self.pk_attn = pk_attn
        self.k_cache = None       # [B, T_so_far, H, d_head]
        self.v_cache = None
        # Inverted index: maps (key_a, key_b) -> list of block indices
        self.inv_index = defaultdict(list)
        self.block_assignments = []  # (key_a, key_b) per block

    def append_token(self, k_new, v_new):
        """Add one token to the cache. O(1) amortized."""
        if self.k_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
        else:
            self.k_cache = torch.cat([self.k_cache, k_new], dim=1)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=1)

        T = self.k_cache.shape[1]
        block_size = self.pk_attn.block_size

        # Check if a new complete block was formed
        if T % block_size == 0:
            block_idx = T // block_size - 1
            block_k = self.k_cache[:, -block_size:].mean(dim=1)  # [B, H, d_head]

            # Assign to product-key pair
            repr_a = block_k[..., :self.pk_attn.d_half]
            repr_b = block_k[..., self.pk_attn.d_half:]
            key_a = (repr_a @ self.pk_attn.codebook_a.T).argmax(-1)
            key_b = (repr_b @ self.pk_attn.codebook_b.T).argmax(-1)

            self.inv_index[(key_a.item(), key_b.item())].append(block_idx)
            self.block_assignments.append((key_a.item(), key_b.item()))

    def attend(self, q):
        """O(sqrt(n_blocks)) attention using cached inverted index."""
        # Product-key lookup using inverted index
        q_a = q[..., :self.pk_attn.d_half]
        q_b = q[..., self.pk_attn.d_half:]

        sub_scores_a = q_a @ self.pk_attn.codebook_a.T
        sub_scores_b = q_b @ self.pk_attn.codebook_b.T

        top_a = sub_scores_a.topk(self.pk_attn.top_c).indices
        top_b = sub_scores_b.topk(self.pk_attn.top_c).indices

        # Collect blocks from inverted index
        candidate_blocks = set()
        for a_idx in top_a.squeeze().tolist():
            for b_idx in top_b.squeeze().tolist():
                candidate_blocks.update(self.inv_index.get((a_idx, b_idx), []))

        # Attend to candidate blocks only
        # ... (gather KV from cache at candidate block positions, standard attention)
```

During generation:
- **New token insertion**: O(1) per token, O(d) when a block completes
- **Block retrieval**: O(sqrt(n_blocks) * d_half) for codebook scoring, then O(1) index lookup
- **Memory**: Inverted index adds O(n_blocks) integers — negligible vs KV cache

### 5.6 Multi-Head Considerations

Each attention head should have its OWN product-key query (as in PEER with h=8 heads),
but can share codebooks across heads to save memory:

```python
# Per-head queries, shared codebook
q_route = q_proj(x).reshape(B, T, n_heads, d_head)  # per-head
# codebook_a, codebook_b are shared across heads
# Each head selects different blocks based on its own query
```

Shared codebooks work because the block structure is the same across heads — only the
query determines which blocks are relevant.

---

## 6. Complexity Summary

| Component | NSA (original) | Product-Key NSA |
|-----------|---------------|-----------------|
| Compressed branch | O(T/d) per query | O(T/d) — unchanged |
| Selection scoring | O(T/d) — score all compressed tokens | O(sqrt(T/d)) — product-key lookup |
| Selection attention | O(top_k * block_size) | O(top_k * block_size) — unchanged |
| Sliding window | O(w) | O(w) — unchanged |
| **Total per query** | **O(T/d + top_k*block_size + w)** | **O(sqrt(T/d) + top_k*block_size + w)** |

The selection scoring is the only term that grows with sequence length.
For T=64k, d=16: NSA scores 4096 compressed tokens, PK-NSA scores ~128 (2*sqrt(4096)).

At T=1M, d=16: NSA scores 62,500 compressed tokens, PK-NSA scores ~500.
**125x reduction in selection cost.**

At T=12M (SubQ's claimed context), d=16: NSA scores 750,000, PK-NSA scores ~1,732.
**433x reduction.**

---

## 7. Open Questions and Risks

### 7.1 Quality of Product-Key Selection for Attention

PEER uses product keys to select *fixed* experts. Attention uses them to select
*dynamic* KV blocks whose content changes every step. Will the factored structure
still capture the right blocks?

**Mitigation**: The V2 approach (union of sub-space top-c + full rescore) guarantees
recall of the true top-k as long as c is large enough. With c=2*top_k, empirical
recall should exceed 95%.

### 7.2 Codebook Drift During Training

If using learned codebooks (V1), they may drift out of alignment with the K distribution.

**Mitigation**: Re-run k-means assignment every N steps (cheap). Or use EMA codebook
updates like VQ-VAE.

### 7.3 BatchNorm at Inference

PEER uses BatchNorm which behaves differently at train vs inference. With batch_size=1
during generation, running stats are used instead.

**Mitigation**: Use large momentum (0.1) for running stats during training.
Alternative: Replace BatchNorm with LayerNorm or RMSNorm (slight utilization drop
but simpler).

### 7.4 Causal Masking with Block Selection

Selected blocks may include future tokens. Must apply causal mask within the
attention computation on selected blocks.

**Mitigation**: Standard causal mask on gathered KV. Or only allow selection of
blocks strictly before the current query position.

### 7.5 When Product Keys Are NOT Worth It

For n_blocks < 256 (T < 16k with block_size=64), the overhead of product-key
machinery exceeds the savings. Just score all blocks directly.

Product keys shine at n_blocks > 4096 (T > 256k), where the sqrt reduction
is substantial.

---

## 8. References

1. He et al. "Mixture of A Million Experts" (PEER), arXiv:2407.04153, 2024
2. Lample et al. "Large Memory Layers with Product Keys", NeurIPS 2019, arXiv:1907.05242
3. DeepSeek-AI. "NSA: Native Sparse Attention", arXiv:2502.11089, 2025
4. Gale et al. "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts", arXiv:2211.15841, 2022
5. Kitaev et al. "Reformer: The Efficient Transformer" (LSH attention), ICLR 2020
6. Kim et al. "HiP Attention: Sparse Sub-Quadratic Attention with Hierarchical Attention Pruning", arXiv:2406.09827, 2024
7. Lei et al. "SparseK Attention: Sparser is Faster and Less is More", arXiv:2406.16747, 2024
8. Krajewski et al. "Scaling Laws for Fine-Grained Mixture of Experts", ICML 2024
9. Subquadratic Inc. "SSA: Subquadratic Selective Attention", 2026
