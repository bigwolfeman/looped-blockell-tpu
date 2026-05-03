# Compaction Incident — K=1 Gate/Up + No Speed Boost + VRAM Increase

**Date**: 2026-05-03
**Run**: pipeline_70k (resumed from step 44k checkpoint with graph-break fix)
**Wandb**: looped-blockell-ablation / pipeline_70k

---

## What Happened

Resumed pipeline_70k from step 44000 checkpoint. CMS score EMAs and prune_round
counter were NOT saved in the checkpoint, so they reset to zero. Two additional
prune rounds fired (step 46k, 49k) with immature scores. At step 50000, compaction
locked in the damage.

### Compaction Results (step 50000)

```
core[0].w_gate: K 32 → 1  (3.1% density)    ← catastrophic
core[0].w_up:   K 32 → 1  (3.1% density)    ← catastrophic
core[0].w_down: K 86 → 3  (3.5% density)
core[1].w_gate: K 32 → 1  (3.1%)
core[1].w_up:   K 32 → 1  (3.1%)
core[1].w_down: K 86 → 4  (4.7%)
core[2].w_gate: K 32 → 1  (3.1%)
core[2].w_up:   K 32 → 1  (3.1%)
core[2].w_down: K 86 → 7  (8.1%)
core[3].w_gate: K 32 → 1  (3.1%)
core[3].w_up:   K 32 → 1  (3.1%)
core[3].w_down: K 86 → 10 (11.6%)
core[4].w_gate: K 32 → 1  (3.1%)
core[4].w_up:   K 32 → 1  (3.1%)
core[4].w_down: K 86 → 13 (15.1%)
core[5].w_gate: K 32 → 1  (3.1%)
core[5].w_up:   K 32 → 1  (3.1%)
core[5].w_down: K 86 → 14 (16.3%)
```

Overall density: 23%. But gate/up are at 3.1% — far below the 25% target.
Pattern: w_down retains more tiles in later blocks. gate/up uniformly K=1.

### Quality Impact

```
step 49500: val_ppl = 27.5   (pre-compact)
step 50000: val_ppl = 287.0  (at compaction — 10x spike)
step 50500: val_ppl = 29.8   (500 steps later — recovered!)
step 51000: val_ppl = 29.5   (1000 steps — stable)
```

Model recovered from catastrophic pruning in ~500 steps. This is remarkable.
Pre-compact val_ppl was 27.5, post-recovery is 29.5 — only 7% degradation
despite gate/up being at 3.1% density (vs intended 25%).

### Speed Impact

```
Pre-compact:  2.7 step/s, 19.2GB VRAM
Post-compact: 2.6 step/s, 22.5GB VRAM
Expected:     3.5+ step/s, ~16GB VRAM
```

**No speed boost from compaction. VRAM increased 3.3GB.**

### Root Cause: CMS Score Reset

The checkpoint does NOT save:
- `prune_round` counter (reset to 0)
- `current_density` (reset to 1.0)
- CMS `score_ema` tensors (these ARE saved in state_dict but reset behavior unclear)

Wait — score_ema IS in the state_dict (`core.0.mlp.w_gate.score_ema: [86, 32]`).
So the scores were loaded correctly. But the prune_round=0 means the `rezero_dead`
call (guarded by `if prune_round > 0`) never fired. Dead tiles' weights could have
drifted via optimizer momentum between step 44k and the prune at 46k.

Actually, the deeper issue: the model was at density 0.32 at step 44k. The tile_mask
was loaded from checkpoint. But the pruning at step 46k pruned 10% of the REMAINING
tiles using only ~2k steps of fresh score accumulation. From 0.32 → 0.26 density
in one round is aggressive — and the immature scores would favor tiles arbitrarily.

With SwiGLU, gate and up projections naturally have lower gradient magnitudes than
down (the SiLU*up gate produces small gradients for gate). So with immature scores,
gate/up tiles are disproportionately pruned.

---

## Two Unsolved Mysteries

### 1. Why No Speed Boost From Compaction?

Previous run (400M): 2.0 → 3.1 step/s at compaction (56% boost).
This run: 2.7 → 2.6 step/s (flat or slightly slower).

Possible explanations:
- **K=1 makes Block-ELL einsum LESS efficient than dense**: gather/scatter overhead
  per tile is fixed, but with K=1 there's only one tile of useful work per row.
  The einsum `x_gathered = x[:, :, col_indices, :]` with K=1 is just a gather of
  16 dims — the indexing overhead dominates the 16x16 matmul.
- **torch.compile may not fuse the Block-ELL einsum well**: the dynamic K per module
  and the col_indices gather may cause compile guards / recompilation.
- **The 400M run used different Block-ELL path**: might have used the Triton kernel
  instead of the einsum path. Need to check.

### 2. Why VRAM Increased 3.3GB?

Pre-compact: 19.2GB. Post-compact: 22.5GB.
The model went from 66.6M → 54.6M params. VRAM should DECREASE.

Possible explanations:
- **torch.compile recompilation**: `torch._dynamo.reset()` + recompile allocates new
  graph buffers. The old compiled graph's memory isn't freed immediately.
- **Block-ELL einsum intermediate tensors**: `x_gathered = x[:, :, col_indices, :]`
  creates a gathered tensor [B, S, R, K, 16] even with K=1. The compile graph may
  keep these intermediates alive for backward.
- **Optimizer rebuild**: new AdamW with fresh state. Old optimizer's m1/m2 tensors
  may not be freed before new ones are allocated.
- **Memory fragmentation**: the compaction changes tensor shapes, causing allocator
  fragmentation.

---

## Action Items

1. **Fix checkpoint format**: save prune_round, current_density in ckpt dict
2. **Fix rezero_dead on resume**: compute prune_round from density on load
3. **Investigate K=1 efficiency**: profile Block-ELL einsum at K=1 vs K=8
4. **Investigate VRAM increase**: compare memory snapshots pre/post compact
5. **Consider**: is K=1 gate/up + K=3-14 down a viable architecture? The 7%
   quality cost for 97% gate/up FLOP savings is remarkable if intentional.

## Backup Checkpoint

```
checkpoints/ablation/pipeline_70k/ckpt_backup_step40k.pt
```

This has the original tile masks at density 0.32 with mature CMS scores.
Can restart from here with the graph-break fix and proper prune_round tracking.

---

## Key Insight: Extreme Pruning Robustness

Despite K=1 on gate/up (3.1% density), the model recovered to within 7% of
pre-compact quality in 500 steps. This suggests:

- The looped architecture (6 blocks × 8 iterations) compensates for crippled MLPs
  by letting attention layers redistribute information between loop iterations
- w_down retaining more tiles (K=3-14) preserves the output projection quality
- Neural memory provides supplemental capacity outside the MLP path
- This may be evidence that gate/up can be pruned much more aggressively than down

This warrants a deliberate ablation: prune gate/up to 5-10% and down to 25%,
with PROPER CMS scores, and compare against uniform 25% pruning.
