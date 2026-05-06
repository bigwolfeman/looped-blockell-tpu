# Hash-Based Sparse Attention: Research Survey

**Date**: 2026-05-05
**Purpose**: Reverse-engineering SubQ's likely attention mechanism. Survey of LSH-based attention methods, failure modes, and implementation considerations for building candidate implementations.

---

## 1. Reformer (Kitaev et al., ICLR 2020)

**Paper**: [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)

### Hash Function: Angular LSH via Random Rotations

Reformer uses angular locality-sensitive hashing based on random rotations of spherically projected points. The specific hash function comes from [Practical and Optimal LSH for Angular Distance (Andoni et al., 2015)](https://arxiv.org/abs/1509.02897).

The hash is computed as:
```
h(x) = argmax([xR; -xR])
```
where `R` is a random rotation matrix of shape `[d_k, n_buckets/2]`. The concatenation `[xR; -xR]` produces `n_buckets` values, and the argmax assigns the token to one of `n_buckets` buckets. This is essentially SimHash with multi-dimensional output -- tokens whose projections align on the same random direction end up in the same bucket.

### Bucket Sizing

- **`num_buckets`**: Should be approximately `seq_len / chunk_length`. For a 4096-length sequence with chunk_length=64, that's ~64 buckets.
- **Factorized buckets**: For very long sequences, `num_buckets` can be factorized as `(n1, n2)` to reduce memory. Each token gets a 2D bucket ID `(b1, b2)` instead of a flat ID.
- **Adaptive**: When `num_buckets=None`, the HuggingFace implementation auto-calculates based on sequence length.
- **NOT fixed-size buckets**: Buckets are inherently variable-size since hashing is random. This is a major GPU efficiency problem.

### Multi-Round LSH

- **`num_hashes`** (default: 1, typically 2-8): Number of independent hashing rounds with different random `R` matrices.
- **Combination strategy: UNION**. A query attends to a key if they share a bucket in ANY round. More rounds = better recall but more computation.
- The multi-round outputs are combined by computing attention separately per round, then averaging (with appropriate normalization via logsumexp).

### Causal Masking Within Buckets

After hashing, tokens are sorted by bucket ID. Within each bucket, they are further sorted by original position. Causal masking is applied by masking out future positions within each sorted chunk. Specifically:
- Tokens attend within their chunk and `lsh_num_chunks_before` previous chunks (default: 1)
- `lsh_num_chunks_after` is set to 0 for causal models
- Within a chunk, standard causal masking by position index is applied

### The Shared-QK Constraint

**Critical limitation**: Reformer ties Q = K (shared projection). This is required because both queries and keys must hash to the same buckets -- if Q and K used different projections, a query's hash wouldn't correspond to the keys it should attend to.

The implication: `attn(q, k) = attn(k, k)`, so attention is purely based on key similarity, not the query-key interaction that standard attention learns. This fundamentally limits expressivity.

### Complexity

- Theoretical: O(n log n) per round, O(n * num_hashes * chunk_length) in practice
- The log n comes from sorting tokens by bucket ID
- Constant factors from hashing, sorting, and chunk management are significant

### Why It Failed to Scale

1. **Shared Q=K kills expressivity**: Can't learn independent query-key interactions
2. **Variable bucket sizes**: Poor GPU utilization -- some buckets overflow, others are near-empty. Padding wastes compute.
3. **Hash quality degrades in high dimensions**: Angular LSH collision probability depends on angle between vectors. In high-d spaces (d_k=64-128), most vectors are nearly orthogonal, so hash discrimination is poor.
4. **Overhead dominates at moderate lengths**: Below ~8k tokens, the hashing/sorting/reindexing overhead exceeds the quadratic attention it replaces.
5. **No learned routing**: Random projections can't adapt to the actual attention patterns the model needs.
6. **Training instability**: Hash assignments change between forward passes (different random R each time), creating noisy gradients.

---

## 2. SMYRF (Daras et al., NeurIPS 2020)

**Paper**: [SMYRF: Efficient Attention using Asymmetric Clustering](https://proceedings.neurips.cc/paper/2020/hash/47d40767c7e9df50249ebfd9c7cfff77-Abstract.html)

### Key Innovation: Asymmetric Hashing

SMYRF's main advance over Reformer: **Q and K can be different**. It uses asymmetric transformations derived from the ALSH (Asymmetric LSH) framework for Maximum Inner Product Search (MIPS).

The core idea: transform the MIPS problem (finding keys with highest dot product to a query) into a nearest-neighbor problem that standard LSH can solve. This is done by applying different transformations to queries vs keys before hashing:
```
P(q) = [q; ||q||^2; ||q||^4; ...; ||q||^(2^m)]     # query transform
Q(k) = [k; 1/2; 1/2; ...; 1/2]                       # key transform
```
After transformation, `<P(q), Q(k)>` approximates the original inner product `<q, k>`, and standard angular LSH on the transformed vectors correctly identifies high-attention pairs.

### Balanced Clustering

SMYRF enforces **balanced buckets** -- every bucket gets the same number of queries AND the same number of keys. This is critical for GPU efficiency (no variable-size padding needed).

Balancing is achieved by sorting tokens by their hash values and then assigning fixed-size groups sequentially, rather than using hard bucket boundaries. This adaptive boundary scheme ensures uniform load.

### Quality vs Speed

- **Drop-in replacement**: Works on pretrained models with no retraining (unlike Reformer which needs shared-QK)
- **GLUE benchmark**: SMYRF-BERT 82.98 vs BERT 82.69 with 50% less memory
- **BigGAN**: FID 25.03 vs 26.06 standard at 128x128
- **Complexity**: O(N log N) same as Reformer, but better constants due to balanced buckets

### Limitations

- Still uses random hashing (not learned)
- The asymmetric transform adds dimensions, increasing hash computation cost
- Balanced clustering via sorting is itself O(N log N)

---

## 3. Routing Transformer (Roy et al., TACL 2020)

**Paper**: [Efficient Content-Based Sparse Attention with Routing Transformers](https://arxiv.org/abs/2003.05997)

### Online K-Means for Bucket Assignment

Instead of random hashing, uses **learned centroids** via mini-batch k-means:
- Maintain `k = sqrt(n)` centroid vectors as model parameters
- Each forward pass: assign tokens to nearest centroid
- Update centroids with exponential moving average (online k-means)
- Tokens in the same cluster attend to each other

### Balanced Routing

Rather than hard Voronoi assignment (which creates variable-size clusters):
- Sort tokens by distance to each centroid
- Take top-`(n/k)` tokens per centroid
- This guarantees balanced clusters of size `sqrt(n)`

### Learned vs Random: The Comparison

- **Wikitext-103**: Routing 15.8 PPL vs Reformer ~18.3 PPL
- **ImageNet-64**: 3.43 bits/dim vs 3.44
- Learning cluster assignments is definitively better than random hashing
- But k-means has **extremely slow convergence** -- centroids take many steps to stabilize

### Gradient Flow

Gradient does NOT flow through the k-means assignment (it's a discrete argmin). The routing is treated as a fixed assignment for each forward pass. Centroids are updated via the k-means update rule, not backpropagation.

### Complexity

O(n^{1.5} * d): The n^{1.5} comes from k = sqrt(n) clusters each of size sqrt(n), so each token attends to sqrt(n) others.

### Limitations

- Centroids converge slowly, especially early in training
- No gradient through routing = suboptimal routing decisions
- The Q=K constraint is partially present (centroids are shared for Q and K assignment)

---

## 4. ScatterBrain (Chen et al., NeurIPS 2021) & KDEformer (Zandieh et al., ICML 2023)

### ScatterBrain: Sparse + Low-Rank Unification

**Paper**: [Scatterbrain: Unifying Sparse and Low-rank Attention Approximation](https://arxiv.org/abs/2110.15343)

Key insight: sparse (LSH) and low-rank (random features/Performer) approximations are complementary:
- **Sparse (LSH)**: Good when attention is "spiky" (few large entries dominate)
- **Low-rank (kernel)**: Good when attention is "diffuse" (many small entries matter)
- **Combined**: Use LSH to capture large entries, kernel approximation for the low-rank remainder

The estimator is **unbiased** with provably lower error than either method alone.

Results: 2.1x lower error than baselines as drop-in replacement in BigGAN and T2T-ViT.

### KDEformer: Kernel Density Estimation + LSH

**Paper**: [KDEformer: Accelerating Transformers via Kernel Density Estimation](https://arxiv.org/abs/2302.02451)

Reduces the softmax denominator computation to a KDE problem:
1. Use LSH to find heavy-hitter entries in the attention matrix
2. Use KDE-based sampling for the remaining entries
3. Provides **spectral norm bounds** (not just entry-wise), which is stronger

Results: 18x speedup on ImageNet classification with <0.5% accuracy drop.

### Relevance to SubQ

These papers show that hash-based selection of "important" attention entries is sound -- the question is efficiency of the selection mechanism itself. Both methods still use linear random hashing, which has the cone/orthogonality problem described below.

---

## 5. MagicPIG (ICLR 2025 Spotlight) & Spotlight Attention (NeurIPS 2025)

### MagicPIG: LSH Sampling for LLM Generation

**Paper**: [MagicPIG: LSH Sampling for Efficient LLM Generation](https://arxiv.org/abs/2410.16179)

The most recent serious attempt at LSH attention for production LLMs:
- Uses **SimHash** (random projections): `h(x) = sign(Rx)` where R is random
- **CPU-offloaded hash tables**: Hash tables + reduced attention on CPU, only final aggregation on GPU
- Achieves 1.5-5x throughput improvement on A100/L20/RTX4090
- 54ms decoding latency for Llama-3.1-8B at 96K context
- <2% accuracy degradation on moderate-to-long context tasks

**Critical finding**: Requires **720-1024 bit hash codes** per token for acceptable retrieval precision with linear hashing. This is because queries and keys in trained LLMs form nearly orthogonal cones in embedding space, making random hyperplane partitioning inefficient.

### Spotlight Attention: Learned Non-Linear Hashing

**Paper**: [Spotlight Attention: Non-linear Hashing-based KV Cache Retrieval](https://arxiv.org/abs/2508.19740)

The breakthrough that fixes linear hashing's fundamental problem:

**Architecture**: 2-layer MLP hasher per head per layer
```
h(x) = sign(W2 * SiLU(W1 * x + b1))
```
- Input dim: 128, hidden dim: tunable, output: 128-bit hash codes
- Trained with Bradley-Terry ranking loss on 8192 samples

**Key results**:
- **5-6x shorter hash codes** than MagicPIG (128 bits vs 720+ bits)
- IoU improvement: 0.18 (linear LSH) -> 0.34 (MLP hashing) for token retrieval
- LLaMA-3-8B on PG19: 8.977 PPL (Spotlight, 98% pruned) vs 8.604 (vanilla) vs 8.881 (oracle top-2%)
- Trainable on a single 16GB GPU in 8 hours

**Why linear hashing fails**: Queries and keys in trained LLMs concentrate in narrow angular cones that are nearly orthogonal to each other. Random hyperplanes cut through these cones randomly, producing hash codes with almost no discriminative power. Non-linear (MLP) boundaries can wrap around the cone structure.

---

## 6. Mixture of Sparse Attention (MoSA, NeurIPS 2025)

**Paper**: [Mixture of Sparse Attention](https://arxiv.org/abs/2505.00315)

This is NOT hash-based but is the strongest modern result and highly relevant to SubQ:

### How It Works

- Each attention head acts as an "expert" that selects its own k tokens via a learned router
- Router: `r = sigmoid(X @ W_r)`, then top-k selection
- Only selected tokens get Q/K/V projections computed
- Attention on k x k matrix instead of T x T
- Unselected positions receive zero output

### Why It Matters

**MoSA is the ONLY sparse attention method that outperforms dense baselines**:
- 28M model: -27% perplexity vs dense at equal compute
- 516M model: -13.3% perplexity vs dense at equal compute
- Fixed sparse: +3.7% worse. Routing Transformer: +3.9% worse.

### Key Insight for SubQ

MoSA shows that **learned, content-dependent token selection BEFORE attention** is strictly better than:
- Random hashing (Reformer)
- Learned clustering (Routing Transformer)
- Fixed patterns (strided/local)

The router selects tokens, not the hash function. This is exactly the regime SubQ's "content-dependent selection" language describes.

---

## 7. Implementation Considerations

### Best Hash Functions for GPU

| Hash Function | Ops | GPU-Friendly? | Quality |
|---|---|---|---|
| **SimHash** (random projection) | One matmul + sign | Excellent (pure GEMM) | Poor in high-d (cone problem) |
| **Cross-polytope** | Random rotation + argmax | Good (Hadamard + argmax) | Better than SimHash |
| **Hyperplane LSH** | Multiple dot products + sign | Good (batched dots) | Same as SimHash |
| **MLP hash** (Spotlight) | 2-layer MLP + sign | Excellent (standard NN ops) | Best (5x shorter codes) |
| **Learned centroids** (Routing) | Distance to K centroids | Good (batched L2) | Good but slow convergence |

**Recommendation**: MLP hashing (Spotlight-style) is the clear winner. It's just a small neural network -- no exotic ops, perfect for GPU, and 5x more efficient than linear hashing.

### Variable Bucket Sizes on GPU

The fundamental tension: LSH produces variable-size buckets, but GPUs want uniform tensor shapes.

**Approaches**:
1. **Padding to max bucket size**: Simple but wasteful. Worst case: one bucket gets most tokens.
2. **Sorted chunking (SMYRF)**: Sort by hash value, take fixed-size chunks. Guarantees balance.
3. **Top-k per bucket (Routing Transformer)**: Each centroid takes exactly top-k tokens. Perfect balance.
4. **Expert-choice (MoSA)**: Each head selects exactly k tokens. No buckets at all.

**Recommendation for GPU**: Avoid variable-size buckets entirely. Use either:
- Sorted chunking (SMYRF-style) for hash-based approaches
- Top-k selection (MoSA-style) for learned routing

### Gradient Flow Through Hashing

| Method | Gradients through routing? | Mechanism |
|---|---|---|
| Reformer | No | Random hash, no parameters |
| SMYRF | No | Random hash, no parameters |
| Routing Transformer | No (for assignment) | Centroids updated via k-means, not backprop |
| MoSA | Yes (through router scores) | `diag(r) * A * W_o` -- router values weight outputs |
| Spotlight | Yes (through MLP) | Bradley-Terry ranking loss trains the hasher |

**Key insight**: Gradient flow through routing is essential for quality. MoSA's success comes partly from end-to-end differentiable routing.

**STE (Straight-Through Estimator)**: Could be used to pass gradients through discrete hash assignments, but in practice learned soft routing (MoSA) works better than STE through hard hashing.

### Memory Layout for Bucket-Local Attention

For any hash/cluster-based attention, the memory layout challenge is gathering scattered tokens into contiguous memory for efficient attention computation:

```
# Naive: gather/scatter (bad -- random memory access)
selected = tokens[bucket_indices]  # scattered reads
attn_out = attention(selected)
output[bucket_indices] = attn_out  # scattered writes

# Better: sort-based (coalesced access)
sort_idx = argsort(hash_values)
sorted_tokens = tokens[sort_idx]   # one sorted permutation
# Now chunks of sorted_tokens are contiguous in memory
attn_out = chunked_attention(sorted_tokens, chunk_size=k)
output = attn_out[inverse_sort_idx]  # one inverse permutation
```

The sort-based approach converts O(n) random accesses into 2 permutations + contiguous chunked attention. This is what Reformer and SMYRF do internally.

---

## 8. Why LSH Attention Hasn't Scaled to Frontier

### Failure Mode Taxonomy

1. **The Cone Problem** (geometric): In trained LLMs, Q and K vectors concentrate in narrow, nearly-orthogonal angular cones. Random hyperplane hashing has almost zero discriminative power in this geometry. Spotlight shows you need ~720 random bits vs ~128 learned bits.

2. **Shared Q=K** (expressivity): Reformer forces Q=K to make hashing consistent. This removes the model's ability to learn independent query and key representations -- a massive expressivity loss.

3. **Approximation Error Compounds** (depth): Each layer's attention approximation introduces error. Over 32-128 layers, small per-layer errors compound catastrophically. Dense attention has zero approximation error.

4. **Training vs Inference Asymmetry**: Hash-based methods were designed for inference efficiency. During training, you need stable gradients, but hash assignments change each forward pass (different random projections), creating high-variance gradient estimates.

5. **The Overhead Crossover** (engineering): Hashing + sorting + gathering + scattering has significant constant-factor overhead. It only wins at very long sequences (>8K-16K tokens), but most training is done on shorter sequences.

6. **No Gradient Signal for Routing Quality** (optimization): Random hashing can't improve -- it's fixed. Learned routing (Routing Transformer) has very slow convergence. Only MoSA-style differentiable routing achieves quality parity.

7. **Hardware Mismatch**: GPUs are optimized for dense, regular computation patterns. Variable-size buckets, scatter/gather operations, and sorting are all anti-patterns for GPU utilization.

### The Fundamental Tension

LSH attention tries to approximate something (full attention) that we know the exact answer to. Any approximation is strictly worse. The only way sparse attention wins is if the computational savings let you do MORE of something else (more heads, more layers, more tokens) that overcomes the approximation loss.

MoSA proves this is possible: by saving compute on attention, you can afford more heads, and the resulting specialization MORE than compensates for the sparsity. But this requires learned routing, not random hashing.

---

## 9. Recommended First Implementation: Learned Router Attention (MoSA-style)

Based on this survey, **we should NOT implement LSH attention first**. Instead, implement a learned router that selects which KV positions each query attends to -- this is what SubQ is most likely doing based on their "content-dependent selection" language.

However, if we want a hash-based variant specifically (for ablation or because SubQ is using hashing), the best candidate is a **Spotlight-style MLP hasher integrated into NSA-like architecture**.

### Pseudocode: Hybrid NSA + Learned Hash Routing

```python
class LearnedHashAttention(nn.Module):
    """
    Combines:
    - Spotlight-style MLP hashing for coarse candidate selection
    - NSA-style compression for global context
    - Local sliding window for fine-grained nearby context
    """
    def __init__(self, d_model, n_heads, d_head, 
                 hash_bits=128, n_tables=4, 
                 local_window=512, top_k_ratio=0.02):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.n_tables = n_tables
        self.top_k_ratio = top_k_ratio
        self.local_window = local_window
        
        # Standard Q, K, V projections (NO shared Q=K)
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)
        
        # Per-head MLP hashers (Spotlight-style, lightweight)
        # One for queries, one for keys (asymmetric)
        self.q_hasher = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_head, d_head, bias=True),
                nn.SiLU(),
                nn.Linear(d_head, hash_bits, bias=False)
            ) for _ in range(n_heads)
        ])
        self.k_hasher = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_head, d_head, bias=True),
                nn.SiLU(),
                nn.Linear(d_head, hash_bits, bias=False)
            ) for _ in range(n_heads)
        ])
        
        # NSA-style compression gate
        self.compress_ratio = 4
        self.compress_proj = nn.Linear(d_head, d_head, bias=False)
        
        # Gating between local, hash-selected, and compressed
        self.gate = nn.Linear(d_head, 3, bias=False)
    
    def _hash_select(self, q, k, v, head_idx, causal_mask_pos):
        """Select top-k keys per query via learned hashing."""
        B, T, D = q.shape
        top_k = max(1, int(T * self.top_k_ratio))
        
        # Compute hash codes (soft, for gradient flow)
        q_hash = self.q_hasher[head_idx](q)  # [B, T, hash_bits]
        k_hash = self.k_hasher[head_idx](k)  # [B, T, hash_bits]
        
        # Hamming similarity via dot product of sign-approximated codes
        # Use tanh as soft sign for gradient flow (STE alternative)
        q_soft = torch.tanh(q_hash * 5.0)  # steep tanh ~ sign
        k_soft = torch.tanh(k_hash * 5.0)
        
        # Hash similarity: higher = more likely to be relevant
        # [B, T_q, T_k] but computed efficiently via matmul
        hash_sim = torch.bmm(q_soft, k_soft.transpose(-1, -2))
        
        # Apply causal mask before selection
        causal = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        hash_sim.masked_fill_(causal.unsqueeze(0), float('-inf'))
        
        # Select top-k positions per query
        _, top_indices = hash_sim.topk(top_k, dim=-1)  # [B, T, top_k]
        
        # Gather selected keys and values
        # top_indices: [B, T, top_k] -> expand for gathering
        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        k_selected = k.unsqueeze(1).expand(-1, T, -1, -1)
        k_selected = torch.gather(k_selected, 2, idx_expanded)  # [B, T, top_k, D]
        v_selected = v.unsqueeze(1).expand(-1, T, -1, -1)
        v_selected = torch.gather(v_selected, 2, idx_expanded)  # [B, T, top_k, D]
        
        # Standard attention over selected subset
        scores = torch.einsum('btd,btnd->btn', q, k_selected) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('btn,btnd->btd', attn, v_selected)
        
        return out
    
    def _local_attention(self, q, k, v):
        """Standard sliding window attention."""
        B, T, D = q.shape
        w = self.local_window
        
        # Efficient: only compute attention within window
        # (use existing windowed attention implementation)
        scores = torch.einsum('btd,bsd->bts', q, k) / (D ** 0.5)
        
        # Mask outside window + causal
        positions = torch.arange(T, device=q.device)
        mask = (positions.unsqueeze(1) - positions.unsqueeze(0))
        mask = (mask < 0) | (mask >= w)  # outside window or future
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum('bts,bsd->btd', attn, v)
    
    def _compressed_attention(self, q, k, v):
        """NSA-style compressed global context."""
        B, T, D = q.shape
        r = self.compress_ratio
        
        # Pool KV into compressed tokens
        T_c = T // r
        k_comp = k.reshape(B, T_c, r, D).mean(dim=2)
        v_comp = v.reshape(B, T_c, r, D).mean(dim=2)
        k_comp = self.compress_proj(k_comp)
        
        # Standard attention over compressed tokens
        scores = torch.einsum('btd,bsd->bts', q, k_comp) / (D ** 0.5)
        
        # Causal mask for compressed positions
        q_pos = torch.arange(T, device=q.device)
        k_pos = torch.arange(T_c, device=q.device) * r + r - 1
        causal = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
        scores.masked_fill_(causal.unsqueeze(0), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum('bts,bsd->btd', attn, v_comp)
    
    def forward(self, x, attention_mask=None):
        B, T, _ = x.shape
        
        q = self.W_q(x).reshape(B, T, self.n_heads, self.d_head)
        k = self.W_k(x).reshape(B, T, self.n_heads, self.d_head)
        v = self.W_v(x).reshape(B, T, self.n_heads, self.d_head)
        
        outputs = []
        for h in range(self.n_heads):
            q_h = q[:, :, h]  # [B, T, D]
            k_h = k[:, :, h]
            v_h = v[:, :, h]
            
            # Three attention branches
            out_local = self._local_attention(q_h, k_h, v_h)
            out_hash = self._hash_select(q_h, k_h, v_h, h, None)
            out_comp = self._compressed_attention(q_h, k_h, v_h)
            
            # Learned gating
            gate_logits = self.gate(q_h)  # [B, T, 3]
            gates = torch.softmax(gate_logits, dim=-1)
            
            out_h = (gates[..., 0:1] * out_local + 
                     gates[..., 1:2] * out_hash + 
                     gates[..., 2:3] * out_comp)
            outputs.append(out_h)
        
        out = torch.stack(outputs, dim=2).reshape(B, T, -1)
        return self.W_o(out)
```

### Why This Design

1. **MLP hashers, not random projections**: Learned non-linear hash functions are 5x more efficient than SimHash (Spotlight result)
2. **Asymmetric Q/K hashing**: Separate hashers for Q and K -- no shared-QK constraint
3. **Soft hashing for gradients**: `tanh(5x)` approximates `sign(x)` but allows gradient flow
4. **Three-branch NSA-style**: Local window + hash-selected sparse + compressed global, with learned gating
5. **Fixed top-k selection**: No variable-size buckets -- always select exactly k positions per query

### Ablation Plan

To determine what SubQ is actually doing, run these ablations:

| Variant | Description | Tests |
|---|---|---|
| `dense` | Full attention baseline | Quality ceiling |
| `local_only` | Sliding window only | Speed ceiling, quality floor |
| `hash_linear` | SimHash (random projections) | Classic Reformer-style |
| `hash_mlp` | Learned MLP hasher | Spotlight-style |
| `router_topk` | MoSA-style learned router (no hashing) | Best quality candidate |
| `hybrid_hash+comp` | Hash selection + NSA compression | Most likely SubQ architecture |
| `hybrid_router+comp` | Router selection + NSA compression | Alternative SubQ candidate |

---

## 10. Key Takeaways

1. **Random hashing is dead for frontier attention**. SimHash/cross-polytope LSH fails because Q/K vectors form narrow orthogonal cones. You need either learned hashing (Spotlight MLP) or learned routing (MoSA top-k).

2. **The Q=K constraint is unacceptable**. Reformer's shared projection kills expressivity. Any viable method must support independent Q and K.

3. **Variable buckets are a GPU anti-pattern**. Use sorted chunking (SMYRF) or fixed top-k selection (MoSA) instead.

4. **Gradient flow through routing is essential**. MoSA's differentiable routing is the only sparse attention that beats dense. Non-differentiable methods (Reformer, Routing Transformer) plateau below dense quality.

5. **The most likely SubQ architecture** is a hybrid: learned token selection (router or hash) for sparse global attention + local window + compressed global context, gated per-head. This matches their "content-dependent selection" language and their claimed 62.5x reduction.

6. **First implementation priority**: MoSA-style router attention with NSA compression branch. If SubQ is using hashing specifically, add Spotlight-style MLP hashers as an alternative selection mechanism.

---

## Sources

- [Reformer: The Efficient Transformer (Kitaev et al., 2020)](https://arxiv.org/abs/2001.04451)
- [SMYRF: Efficient Attention using Asymmetric Clustering (Daras et al., 2020)](https://proceedings.neurips.cc/paper/2020/hash/47d40767c7e9df50249ebfd9c7cfff77-Abstract.html)
- [Routing Transformer (Roy et al., 2020)](https://arxiv.org/abs/2003.05997)
- [ScatterBrain (Chen et al., 2021)](https://arxiv.org/abs/2110.15343)
- [KDEformer (Zandieh et al., 2023)](https://arxiv.org/abs/2302.02451)
- [MagicPIG: LSH Sampling for LLM Generation (ICLR 2025)](https://arxiv.org/abs/2410.16179)
- [Spotlight Attention: Non-linear Hashing (NeurIPS 2025)](https://arxiv.org/abs/2508.19740)
- [Mixture of Sparse Attention / MoSA (NeurIPS 2025)](https://arxiv.org/abs/2505.00315)
- [Practical and Optimal LSH for Angular Distance (Andoni et al., 2015)](https://arxiv.org/abs/1509.02897)
- [Reformer HuggingFace Documentation](https://huggingface.co/docs/transformers/en/model_doc/reformer)
- [lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)
- [lucidrains/routing-transformer](https://github.com/lucidrains/routing-transformer)
- [Infini-AI-Lab/MagicPIG](https://github.com/Infini-AI-Lab/MagicPIG)
