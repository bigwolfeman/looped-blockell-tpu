# NES-Ternary Experiment Plan — 2026-05-04

## Context

EGGROLL (ES at Hyperscale, arXiv:2511.16652) fundamentally cannot work for ternary {-1,0,1} weights because ±1 step = 100% of weight range (vs 0.4% for int8). Verified empirically: diverges at every hyperparameter setting we tried.

We're pivoting to NES-Categorical: maintain a probability distribution over {-1,0,1} per weight, optimize the distribution parameters (continuous logits) using population-based ES with score-function gradients. The ternary weights are SAMPLES from the distribution.

Key innovation: **block-coordinate cycling with momentum** — partition params into groups, cycle through them, momentum carries signal for inactive groups. Reduces required pop from O(P) to O(P/G) per step.

## Model

All experiments use d=128, n_layers=3, d_ff=344 (~13.2M total params, ~118k linear params at 20% density after pruning). Phase 1 checkpoint exists: `checkpoints/ab_phase1_d128L3.pt` (trained STE+Adam, 5000 steps).

## Baseline (Already Run)

**STE+Adam at 20% density**: val_loss=5.48, val_ppl=239 after 5000 steps in 46s.
This is the target to match.

## Experiment Matrix

### Group A: Pure NES-Categorical (Score Function / REINFORCE)

| ID | Method | Groups | Pop/group | Momentum | Tau schedule | Key question |
|----|--------|--------|-----------|----------|--------------|-------------|
| A1 | Full-space NES | 1 (all 118k) | 2048 | None | cos 2.0→0.05 | Does NES work at all with enough pop? |
| A2 | Full-space NES + momentum | 1 | 2048 | 0.9 | cos 2.0→0.05 | Does momentum help full-space? |
| A3 | Per-matrix cycling | 21 | 256 | None | cos 2.0→0.05 | Does cycling reduce required pop? |
| A4 | Per-matrix cycling + momentum | 21 | 256 | 0.9 | cos 2.0→0.05 | **Main hypothesis**: cycling + momentum |
| A5 | Per-layer cycling + momentum | 3 | 512 | 0.9 | cos 2.0→0.05 | Coarser grouping — simpler, faster? |
| A6 | A4 but natural gradient | 21 | 256 | 0.9 (nat) | cos 2.0→0.05 | Does Fisher-corrected gradient help? |

All Group A use:
- Antithetical sampling via CDF-inversion (u and 1-u)
- Fitness shaping (rank-based utilities)
- Baseline subtraction (running EMA)
- 2-DOF logit parameterization (theta_neg, theta_pos; theta_zero=0)

### Group B: LR-nets Reparameterization (CLT-based, uses backprop on distribution params)

| ID | Method | Notes |
|----|--------|-------|
| B1 | LR-nets (Shayer et al. 2018) | Maintain per-weight (p_neg, p_zero, p_pos), use CLT to compute pre-activation mean/var, backprop through reparameterized Gaussian. NO score function, NO population. |
| B2 | LR-nets + entropy regularization | Add entropy penalty to prevent distribution collapse |

Key difference from Group A: uses BACKPROP through the CLT approximation. Not gradient-free. But optimizes distribution params, not shadow weights — so avoids the STE ternary floor.

### Group C: Hybrid Approaches

| ID | Method | Notes |
|----|--------|-------|
| C1 | STE Phase 1 → NES-Cat Phase 3 (A4) | Use STE to find structure, then NES-Cat for discrete optimization. The pipeline Wolfe originally designed. |
| C2 | STE Phase 1 → NES-Cat with warm start | Initialize logits from STE's ternary values: θ_k = 10 if w=k, else -10 (peaked at STE's answer). NES fine-tunes. |
| C3 | NES-Cat from scratch | No Phase 1. Train the distribution from uniform initialization. Tests whether NES can discover structure without BPTT. |

### Group D: Controls

| ID | Method | Notes |
|----|--------|-------|
| D1 | STE+Adam baseline at 20% density | Already done: val_loss=5.48 |
| D2 | STE+Adam baseline at 80% density | Already done: val_loss=5.36 |
| D3 | Random search (uniform sampling, no gradient) | Establishes noise floor |

## Shared Settings

- Model: d=128, n_layers=3, d_ff=344, vocab=49152
- Data: OpenWebText streaming, B=4, seq=256
- Steps: 5000 per experiment (extend to 20k if promising)
- Eval: val_loss every 500 steps, 10 batches
- Logging: wandb project `bonsai-ternary-test`
- Prune: 20% density, per-layer, skip embed/lm_head (keep intact)
- Phase 1 checkpoint: `checkpoints/ab_phase1_d128L3.pt`

## Priority Order

1. **A4** (per-matrix cycling + momentum) — main hypothesis, fastest path to result
2. **A1** (full-space NES) — control: does NES work at all?
3. **B1** (LR-nets) — strongest theoretical approach, but uses backprop
4. **C2** (STE → NES warm start) — the pipeline dream
5. Everything else

## Memory Comparison

| Method | Persistent VRAM per param | Notes |
|--------|--------------------------|-------|
| AdamW+STE | 13 bytes (shadow + m1 + m2 + grad) | Current baseline |
| NES-Cat (bare) | 5 bytes (2 logits bf16 + int8 ternary) | No optimizer state! |
| NES-Cat + SGD momentum | 9 bytes (+ momentum buf) | Recommended |
| NES-Cat + Adam | 13 bytes (same as AdamW) | No advantage |
| LR-nets | ~10 bytes (2 dist params + grad + optim) | Depends on optimizer |

## Key Math (from research)

### NES Score Function Gradient (per weight, per logit k)
```
∇_θk = (1/N) Σ_n (L_n - baseline) × (1_{w_n=k} - p_k)
```

### Natural Gradient (advantage form)
```
∇̃_θk = E[L | w=k] - E[L]
```
Fisher is free: F = diag(p) - pp^T. The natural gradient is just the per-state advantage.

### Antithetical Sampling (CDF inversion)
```
u ~ Uniform(0,1), w = CDF_inv(u)
u_anti = 1-u, w_anti = CDF_inv(1-u)
```
Valid for categorical distributions. Creates negative correlation → variance reduction.

### Block-Coordinate Cycling
- G groups of P/G params each
- Pop needed per group: O((P/G) / Δ²) instead of O(P / Δ²)
- Momentum on logits carries signal between cycles
- Convergence: standard block-coordinate descent theory applies (softmax is smooth)

### Population Requirement
- Full space: N = Ω(P/Δ²) where Δ = marginal advantage gap
- Per-matrix (G=21): N_per_group = Ω(P/(21·Δ²)) — 21× smaller pop per step
- With fitness shaping: variance bounded by p_k(1-p_k)/N regardless of loss scale

## Key References

- **Discrete NES**: arXiv:2404.00208 — NES update rules for categorical distributions
- **LR-nets**: arXiv:1710.07739 (ICLR 2018) — probabilistic ternary via CLT reparameterization
- **EGGROLL**: arXiv:2511.16652 — ES at hyperscale (int8, NOT ternary)
- **GFT**: arXiv:2410.09734 — gradient-free ternary training, proves NP-hardness
- **CatCMA**: arXiv:2405.09962 — natural gradient for categorical variables
- **IndeCateR**: NeurIPS 2023 — low-variance categorical gradient estimator

## Files

- `run_ab_eggroll.py` — current A/B test harness (needs NES-Cat trainer added)
- `kernels/eggroll_fast.py` — fast population evaluator (reusable for NES forward passes)
- `eggroll_model.py` — EGGROLL model (basis for NES model)
- `checkpoints/ab_phase1_d128L3.pt` — Phase 1 checkpoint for d=128 model

## Implementation Plan

1. Build `NESCategoricalTrainer` class with:
   - 2-DOF logit parameterization
   - CDF-inversion sampling + antithetical pairs
   - Seed-based sample regeneration (zero persistent pop storage)
   - Block-coordinate cycling support (configurable groups)
   - SGD + momentum on logits
   - Temperature schedule
   - Fitness shaping (rank-based)
   - Running EMA baseline

2. Build `LRNetTrainer` class (for Group B) with:
   - Per-weight distribution params
   - CLT reparameterization (mean + variance of pre-activations)
   - Standard backprop on distribution params

3. Integrate both into `run_ab_eggroll.py` as new arms

4. Run priority experiments: A4 → A1 → B1 → C2
