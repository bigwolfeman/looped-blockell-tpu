# DeepSeek-V4: Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA)

**Paper**: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence" (Apr 24, 2026)
**Report**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
**Models**: V4-Pro (1.6T/49B active), V4-Flash (284B/13B active)

## The Big Idea

V3 used MLA (compressed in **feature dimension** — low-rank KV heads).
V4 compresses in the **sequence dimension** (grouping/pooling tokens).
These are orthogonal — could combine both.

## Two Mechanisms, Interleaved

### Compressed Sparse Attention (CSA)
1. **KV Compression**: Groups of 8 tokens → 1 KV entry via learned softmax-gated pooling, stride=4 → **4× KV reduction**
2. **Lightning Indexer**: FP4 multi-head scorer, selects **top-k=512** compressed KV entries per query (same idea as our DSA LightningIndexer but over compressed tokens)
3. **Sliding window**: Last 128 uncompressed tokens attended directly
4. **Shared KV vectors** across heads (2× memory savings) + inverse RoPE correction

### Heavily Compressed Attention (HCA)
1. **KV Compression**: 128 tokens → 1 entry, stride=128 → **128× KV reduction**
2. **Dense attention** over compressed sequence (1M tokens → ~7,800 entries — cheap)
3. **Sliding window**: Same 128-token window

### Attention Sinks
Learnable sink logits in softmax denominator — total attention mass can be < 1,
so queries don't spread to irrelevant tokens. Trained, not fixed.

### Layer Interleaving (V4-Pro, 61 layers)
```
Layers 0-1:   HCA (broad global context first)
Layers 2-60:  Alternating CSA ↔ HCA
MTP block:    Sliding-window only
```

## Results at 1M Tokens

| Metric | V4-Pro vs V3.2 |
|--------|----------------|
| Inference FLOPs | **27%** of V3 |
| KV cache memory | **10%** of V3 |
| KV cache at 1M | 9.62 GiB (~2% of GQA-8 baseline) |

## NSA → CSA Evolution

NSA (arXiv:2502.11089, Feb 2025) was the research prototype:
- "Dynamic hierarchical sparse strategy: coarse token compression + fine token selection"
- CSA is NSA productionized with Lightning Indexer (FP4 scorer) replacing NSA's selection

**NSA was the paper. CSA is the product.**

## How V4's Attention Differs from Our DSA

| Feature | Our DSA | V4 CSA |
|---------|---------|--------|
| Compression | Mean-pool blocks | Learned softmax-gated pooling |
| Indexer | Block-level scoring → expand | Per-query top-k over compressed KV |
| Granularity | Block-of-tokens | Individual compressed entries |
| Window | None | 128-token sliding window |
| Sinks | No | Learnable attention sinks |
| HCA fallback | No | Ultra-compressed dense path |

## Relevance to Our Architecture

1. **CSA "compress-then-select" ≈ Block-ELL "prune-then-route"** — same paradigm,
   applied to attention instead of MLP. The Lightning Indexer is conceptually identical
   to our BlockRouter.

2. **Could unify sparsity** — same importance scoring for MLP tile pruning AND
   attention KV selection. CMS gradient norms could inform both.

3. **Looped transformer benefit** — compression reduces per-iteration attention cost.
   With T=6 loops, 4× KV reduction means ~4× cheaper attention per loop iteration.

4. **HCA for outer SSM context** — the ultra-compressed representation (128× reduction)
   is a natural fit for cross-sequence state. The outer SSM state could attend over
   HCA-compressed history from previous sequences.

5. **mHC is in V4 too** — V4 uses manifold-constrained hyper-connections for residuals.
   See `mHC-hyper-connections.md`.

## Implementation for JAX

1. **KV Compressor**: 1D learned pooling over token windows — `jnp.einsum` or `jax.lax.conv_general_dilated`
2. **Lightning Indexer**: Multi-head dot-product scorer + `jax.lax.top_k` — same pattern as our DSA
3. **Sparse gather**: `jnp.take` selected KV, standard attention over small gathered set
4. **Sliding window**: Causal mask with window cutoff
5. **Sinks**: `jnp.concatenate([logits, sink_params], axis=-1)`, softmax, discard sink dim
6. **Inverse RoPE**: RoPE with negated angles on attention output
7. **FP4 indexer**: Use bf16 on TPU (no FP4 support), negligible quality difference

## Also Uses

- **Muon optimizer** (Newton-Schulz orthogonalization)
- **On-Policy Distillation** from 10+ domain specialists
- **MoE with hash-routing** + `sqrt(softplus(.))` gating
- **32T+ training tokens**, FP4+FP8 mixed precision
