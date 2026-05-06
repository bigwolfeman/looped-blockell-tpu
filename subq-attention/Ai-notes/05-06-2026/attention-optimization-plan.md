# Attention Optimization Plan — 2026-05-06

## SubQ Ablation Results (so far)

| Method | bs | Val PPL @ 6k | vs Dense | Complexity |
|---|---|---|---|---|
| Dense | 8 | 43.9 | baseline | O(n²) |
| Dense | 4 | 53.2 | token-matched | O(n²) |
| NSA | 4 | 54.6 | +2.6% | O(n²/r) |
| NSA+MoSA | 8 | 43.3 | -1.4% | O(n²/r + nk) |
| NSA+Cosine | 4 | running | TBD | O(n²/r + nk) |

**Key finding**: MoSA router replaces quadratic selection with zero quality loss. NSA matches dense within 3% at equal tokens.

**Complexity note**: NSA+MoSA is still O(n²/r) due to compressed branch. Pure MoSA+window = O(nk + nw) truly linear.

## Attention Optimizations to Implement

### Priority 1: Core efficiency
- **GQA** — Share KV heads (n_kv < n_heads). Standard KV cache reduction.
- **QK-Norm** — Normalize Q,K before dot product. Training stability at scale.
- **MLA** — DeepSeek V2/V3 low-rank KV compression. Most aggressive KV savings. Requires absorbing RoPE into projection.

### Priority 2: Architecture quality
- **XSA** — Exclude self from attention. Prevent identity shortcut.
- **Residual Attention** — out = attn(x) + α·x. Gradient flow guarantee. Proven in ternary model.
- **CLA** — Cross-Layer Attention. Cache KV on loop iter 0, reuse for 1+. Critical for looped core.

### Priority 3: Architecture search
- **Per-layer mixing gate** — Learned gate blending dense vs MoSA per layer. Discovers optimal attention pattern.
- **RoPE vs CoPE** — CoPE (arXiv 2602.05258) soft-clips low-frequency RoPE for better length extrapolation. +10.84% within training range, ~2x at 256k. Zero inference overhead.

### Final integration
- **Block-ELL + subquadratic attention** — Sparse MLPs (25% density, macro-block execution) + sparse attention (MoSA/NSA). Target: 200-400x total FLOP reduction per layer.

## CoPE (Clipped RoPE) Details

From arXiv 2602.05258:
- Soft-clips low-frequency RoPE components using cosine-decay taper
- Frequencies with periods > pretraining context length → gradually attenuated
- Avoids spectral leakage (Gibbs phenomena) from hard clipping
- Critical dimension: d_ct = 2⌈(d/2)·log_b(L_pre/2π)⌉
- Plug-and-play: modify frequency weights at init, no architecture change
- Results: +10.84% HELMET avg within 64k, ~2x at 256k extrapolation

## Excluded (deprioritized)

- **Differential Attention** — noise cancellation via dual attention maps. Adds complexity, unclear benefit with MoSA routing.
- **Attention Sinks** — learnable sink logits. MoSA routing partially subsumes (router can assign low scores = "nothing relevant").
