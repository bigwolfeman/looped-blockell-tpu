# Architecture Uplift Tasks

## Efficiency Upgrades (implement before next long run)

### 1. SwiGLU Activation
- Replace GELU MLP with SwiGLU: `gate = silu(W_gate(x)), out = W_down(gate * W_up(x))`
- Change d_ff from 4×d to (8/3)×d rounded to tile_size to match param count
- Affects: `interop/pt_model.py` MLPBlock, `looped_blockell/layers/mlp.py`
- Must work with BELL sparse format (same structure, just different activation)
- ~1-2% PPL improvement, zero compute cost

### 2. QK-Norm
- RMSNorm on Q and K after projection, before RoPE
- `q = norm_q(q)`, `k = norm_k(k)` (separate learned scales)
- Affects: `interop/pt_model.py` MultiHeadAttention, `looped_blockell/layers/attention.py`
- Stabilizes attention logits, especially important with memory + AttnRes + XSA
- Near-zero compute cost

### 3. Cross-Layer Attention (CLA) in Core Loop
- Compute K, V once at loop entry (iteration 0)
- Reuse cached K, V for iterations 1-5 (only recompute Q)
- Only applies to core blocks, not prelude/coda
- ~5× reduction in KV compute for the core loop
- Affects: LoopedTransformerPT.forward(), core block attention
- Design: cache KV after first iteration, pass as args to subsequent iterations

### 4. GQA Ratio for Core Blocks
- Reduce n_kv_heads in core blocks (currently n_kv_heads = n_heads)
- Prelude: full MHA (diverse feature extraction)
- Core loop: GQA 4:1 (e.g., 12 Q heads, 3 KV heads at d=768)
- Coda: full MHA or GQA (TBD)
- Combined with CLA: massive savings in looped attention
- Config: add `n_kv_heads` field, default = n_heads (backward compat)

## Remaining Ablations

### 5. Multi-Token Prediction (MTP=3)
- 3 prediction heads sharing backbone
- Output projections: dense (not BELL sparse)
- Need to consider MTP + BELL interaction for training loss
- Reference: Meta MTP paper

### 6. ReMoE Routing over Macro BELL Tiles
- Iteration-aware routing: different loop iterations activate different tile groups
- Already have foundation from 005-routed-block-ell spec
- Routes over macro tiles (hardware-aligned concatenated BELL tiles)

## Dependency Order
1. SwiGLU (independent, easy)
2. QK-Norm (independent, easy)
3. GQA ratio (needed before CLA)
4. CLA in loop (depends on GQA for max savings)
5. MTP (independent of above)
6. ReMoE (independent, builds on existing routing code)

Items 1-2 can be done in parallel. Items 3-4 are sequential.
Items 5-6 are independent of 1-4.
