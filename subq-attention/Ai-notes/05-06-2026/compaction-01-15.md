# Context Dump — 01:15 2026-05-06

## Current Work
Reverse-engineering subquadratic attention (SubQ/SSA). Built 9 attention modules, ran ablation on 4+cosine. Now implementing attention optimizations for the final architecture.

## Key Files
- `subq-attention/attention.py` — 9 pluggable attention modules (dense, sliding_window, nsa, nsa_lsh, nsa_block_router, lod_attention, mosa, nsa_mosa)
- `subq-attention/attention_cosine.py` — NSA+cosine sim (MegaContext internalized)
- `subq-attention/model.py` — SubQTransformer, 80M params, d=768/12heads/6layers
- `subq-attention/train.py` — training loop + scaling measurement
- `subq-attention/research-nsa.md` — DeepSeek NSA paper deep-dive
- `subq-attention/research-lsh.md` — LSH survey (MoSA is the winner)
- `subq-attention/research-product-keys.md` — PEER product key analysis
- `interop/pt_model.py` — PrunableLinear with macro-block support (NEW)

## Ablation Results (Final)

| Method | bs | Val PPL @ 4k steps | vs Dense | Complexity |
|---|---|---|---|---|
| Dense | 8 | 54.0 | ceiling | O(n²) |
| Dense | 4 | 69.9 | token-matched | O(n²) |
| NSA | 4 | 75.5 | +8% vs dense bs=4 | O(n²/r) |
| NSA+MoSA | 8 | 52.2 | -3.3% vs dense bs=8 | O(n²/r + nk) |
| Cosine | 4 | 70.5 | +0.9% vs dense bs=4 | O(n²/r + nk) |

**Key findings:**
1. NSA matches dense within 3% at equal tokens
2. NSA+MoSA BEATS dense by 1-3% (MoSA = implicit regularizer)
3. Cosine sim matches dense — raw key geometry sufficient for selection
4. NSA+MoSA still O(n²/r) due to compressed branch. Pure MoSA+window = O(n) linear.

## Footguns & Gotchas
- **Causal masking in compressed attention**: Compressed blocks mix future tokens. Must use strict `<` (not `<=`) for block-level causal check. Added NaN-safe uniform fallback for first-block queries.
- **Batch size confound**: Dense bs=8 vs NSA bs=4 showed 33% gap that was entirely data volume, not architecture quality. Always token-match comparisons.
- **LR schedule confound**: Shorter --steps = faster LR decay = unfair comparison. Always use same --steps, kill early if needed.
- **S×S mask materialization**: NSA/block_router/cosine variants materialize full S×S bool masks via Python scatter loops. OOMs at seq≥4096 with bs=8. MoSA avoids this via gather-based selection.
- **Extension module pattern**: attention_cosine.py auto-imports via `_load_extension_modules()` at bottom of attention.py.

## Decisions Made
- MoSA-style learned routing is the primary subquadratic selection method
- Macro-block support added to PyTorch PrunableLinear (32 for 5090, 128 for TPU)
- Column reorder permutes d_ff dimension (rows of fc1, cols of fc2)
- CoPE (Clipped RoPE, arXiv 2602.05258) added to ablation plan for length extrapolation

## Tasks Remaining (#125-#133)
- GQA, QK-Norm, MLA, XSA, Residual Attention, CLA
- Per-layer attention mixing gate
- RoPE vs CoPE ablation
- Block-ELL + subquadratic attention integration

## Context That Won't Survive
- MegaContext repo cloned to /mnt/BigAssDrive/00projects/00DeepNet/111TitanMAC-Standalone/MegaContext/
- wandb project: subq-attention-ablation (runs: dense_baseline, nsa_faithful_v3, nsa_mosa_v3, nsa_cosine_v4, dense_bs4_15k)
- The user is going to bed. Work through tasks autonomously.
- NSA+MoSA compressed branch is still quadratic — the "truly linear" version is pure MoSA+window (already implemented as `mosa` attention type)
