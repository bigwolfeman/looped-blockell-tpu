# EGGROLL Session Handoff — 2026-05-04

## GOAL: Write a Triton kernel for EGGROLL forward pass to push population size from 128 → 10k+

## Current State
- EGGROLL implementation is bug-fixed and correct (3 bugs found and patched)
- Transfer from STE→EGGROLL verified clean (loss 3.37 matches exactly)
- The algorithm WORKS but pop=128 for 69M params = pure noise gradient
- Need pop/param ratio ~1:1000 minimum → need pop=69k for 69M params
- At pop=128: 3.0s/step. At pop=69k serial: ~27 minutes/step. NEED KERNEL.

## The Kernel Needed
**Fused EGGROLL rank-1 perturbed linear forward:**
```
For N population members sharing input x and weight W:
  y_base = x @ W^T                          (compute ONCE, cuBLAS)
  y_i = y_base + sigma * (x @ b_i) * a_i^T  (per-member rank-1 correction)
```
- The base matmul is shared — ONE cuBLAS call for all N members
- The rank-1 correction per member: two cheap ops (GEMV + outer product broadcast)
- The bottleneck without kernel: we recompute `x @ W^T` for every member because hidden states diverge across layers
- The kernel should fuse: base matmul + N rank-1 corrections in one launch

### Why hidden states diverge (the hard part)
After layer 0, each member has different hidden state h_i. So at layer 1, `h_i @ W_1^T` is different per member — can't share the base anymore. Options:
1. **Accept O(N) full forwards** but make each one fast (ternary matmul kernel)
2. **First-order approximation**: pretend corrections are small, share base across layers, add corrections as a perturbation. Valid when sigma is small.
3. **Chunk processing**: batch K members at a time through standard forward (current approach, K=16-32 fits L2)

### L2 Cache Opportunity (5090: 128MB L2)
At 20% pruned density (14M ternary params):
- Weights: 14M × 1 byte = 14MB (L2 resident)
- Activations per chunk (K=32): 32 × 4 × 512 × 512 × 2 = 67MB
- Total: ~81MB < 128MB L2
- Zero DRAM round-trips for core matmuls

## Pipeline: BPTT → Prune → EGGROLL
1. Phase 1: STE+Adam, B=12, 20k steps → val_ppl ~55 ✅ (checkpoint exists: `checkpoints/pipeline_v3_phase1.pt`)
2. Phase 2: Per-layer prune to target density. Cold prune survives at 80%, collapses at 60%. CMS gradual pruning needed for lower density.
3. Phase 3: EGGROLL on pruned model with high pop

## Footguns
1. **Cold pruning kills the model below 80% density.** Must use CMS-style gradual pruning during Phase 1 to go lower. We verified: 80%=OK (loss +0.4), 60%=dead (loss +6).
2. **EggrollEmbedding had wrong update rule** — was doing continuous update instead of thresholded ±1. FIXED but verify if subagent's `train_pipeline.py` transfer code uses the old path.
3. **Sigma must be ~0.001 for ternary weights** (effective weight magnitude ~0.02). Paper used 0.0625 for int8 range [-128,127] = 0.025% of range. Our 0.001/0.02 = 5% is still high but workable.
4. **Divide ES gradient by N (full pop), not N_half.** Fixed in eggroll_model.py line 379.
5. **Attention is the VRAM bottleneck** when batching population: (N×B×H×S×S×2) bytes. At N=256, B=4, H=8, S=1024: 32GB. Must chunk.
6. **train_pipeline.py Phase 1 batch size**: use `--phase1_batch_size 12` (separate from `--batch_size 4` for EGGROLL). B=4 gives val_ppl 105 (bad), B=12 gives val_ppl 55 (good).
7. **The 5.3 loss floor** applies to ALL backprop-through-ternary methods (DQT, flip, QSGD, shadow-drop). EGGROLL is the only approach that sidesteps this because it doesn't use gradients.

## Files
- `eggroll_model.py` — bug-fixed EGGROLL model with mask support
- `train_pipeline.py` — 3-phase pipeline (BPTT→Prune→EGGROLL)
- `train_eggroll.py` — standalone EGGROLL training script
- `train_baseline_pruned.py` — A/B baseline (STE continue after prune)
- `experiments/eggroll_audit.md` — bug audit with line numbers
- `experiments/eggroll_research.md` — paper summary
- `experiments/eggroll_curves.md` — training curve analysis from paper
- `kernels/eggroll_fwd.py` — initial kernel attempt (SLOWER than PyTorch, don't use)
- `checkpoints/pipeline_v3_phase1.pt` — ready Phase 1 checkpoint (val_ppl 54.7)

## Other Models Built This Session (all hit 5.3 floor)
- `dqt_model.py`, `ecpg_model.py`, `qsgd_model.py`, `shadow_drop_model.py`, `lste_model.py`
- These are NOT dead — the state machine approaches (DQT/ECPG) may combine with EGGROLL later
