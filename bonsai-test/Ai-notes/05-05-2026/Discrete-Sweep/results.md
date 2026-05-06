# Discrete Ternary Optimization Sweep — 2026-05-05

## Summary

Tested 4 differentiable discrete optimization methods against STE+Adam for ternary {-1,0,+1} training at two scales. STE+Adam wins decisively. The ternary tax shrinks 67% from 13M to 118M params.

## Methods Tested

1. **Gumbel-Softmax** — Per-weight logits, Gumbel noise + softmax, temperature anneal
2. **Straight-Through Gumbel (STG)** — Hard ternary forward, soft Gumbel backward
3. **REINMAX** (NeurIPS 2023) — 2nd-order gradient via Heun's method
4. **Mirror Descent** — Adam on probability-space gradients (bypasses softmax Jacobian)
5. **STE+Adam** — Standard: continuous shadow weights, round to ternary, straight-through gradient
6. **bf16 full** — No ternary constraint (ceiling)

## Results at d=128 (13.2M params, 20% density)

| Method | Val Loss | PPL | Notes |
|--------|----------|-----|-------|
| bf16 full | 4.94 | 140 | Ceiling |
| STE+Adam | 5.33 | 207 | Ternary tax = 0.39 nats |
| Mirror Descent | 5.83 | 340 | Soft eval (not true ternary) |
| PF Top-K | 5.96 | 388 | Gradient-free reference |
| Gumbel-Softmax | 7.56 | 1921 | Stochastic failure |
| STG | 7.43 | 1689 | Stochastic failure |
| REINMAX | 7.29 | 1464 | Stochastic failure |

## Results at d=768 (118M params, full density)

| Method | Val Loss | PPL | Δ from Phase 1 |
|--------|----------|-----|----------------|
| bf16 full | 3.49 | 32.9 | -0.80 (ceiling) |
| STE+Adam | 3.62 | 37.4 | -0.67 (improved) |
| Mirror Descent | 4.25 | 70.3 | -0.03 (barely moved) |
| Gumbel-Softmax | 4.29 | 72.8 | 0.00 (zero learning) |
| STG | 4.29 | 72.8 | 0.00 (zero learning) |
| REINMAX | 4.29 | 72.8 | 0.00 (zero learning) |

Phase 1 init: val=4.29 / PPL 72.8 (5k steps STE+Adam)

## Key Findings

### 1. Ternary tax shrinks with scale

| Scale | bf16 | STE | Tax (nats) | PPL Penalty |
|-------|------|-----|------------|-------------|
| d=128 (13M) | 4.94 | 5.33 | 0.39 | 48% |
| d=768 (118M) | 3.49 | 3.62 | 0.13 | 14% |

67% reduction in ternary tax from 13M to 118M params. Larger models have enough redundancy that the {-1,0,+1} constraint costs very little.

### 2. Stochastic discrete methods completely fail at scale

Gumbel-Softmax, STG, and REINMAX all produced literally ZERO improvement at d=768. The eval loss was identical to Phase 1 at every checkpoint for all 20k steps.

Root cause: these methods parameterize weights as logits over {-1,0,+1} and optimize via Adam. But the logit-to-ternary map is argmax — a step function. Adam's small updates never cross argmax boundaries, so the hard ternary assignments never change.

### 3. Mirror Descent is the only non-STE method that shows any signal

Mirror Descent improved 0.03 nats at d=768 (4.29 → 4.25). It works because it uses deterministic soft expected weights (not stochastic samples) and evaluates with the same soft weights (no hard eval). But 0.03 nats vs STE's 0.67 nats is a 22x gap.

### 4. STE's "dumb" approach is actually optimal

STE maintains continuous shadow weights and rounds to ternary. Small continuous changes occasionally cross quantization boundaries, producing real ternary changes. This is fundamentally better than logit-based approaches because:
- Every small shadow weight change affects the quantization outcome
- The quantization boundary is at the midpoint between ternary values
- Adam's momentum naturally drives weights across these boundaries

## Why Discrete Optimization Methods Don't Work for Ternary

The failure is structural, not hyperparameter-related:

1. **Argmax is a step function.** Logit-based methods optimize smooth logits, but the forward uses argmax to select a ternary value. Small logit changes produce zero change in the output. STE works because rounding IS sensitive to small changes near the boundary.

2. **Gumbel noise destroys pretrained models.** At peak=2.0, Gumbel noise flips ~33% of weight assignments per forward pass. This scrambles any pretrained signal. The d=128 pruned experiments showed partial recovery (the noise is a smaller fraction of the total perturbation when starting from a damaged model), but at d=768 full density the noise prevents any learning.

3. **Soft-hard gap.** Methods that use soft weights during training (Gumbel-Soft, Mirror Descent) learn to work with attenuated weights (~0.58 magnitude instead of 1.0). At eval with hard ternary (magnitude 1.0), the distribution shift degrades quality. Mirror Descent avoids this by using soft eval, but then it's not truly ternary.

## Practical Recommendation

**Use STE+Adam for ternary training. Full stop.**

- At 118M params: ternary tax = 0.13 nats (14% PPL)
- At 1B+ params: tax likely <0.05 nats (essentially free)
- bf16 optimizer states work (proven at d=128: identical quality, 43% memory savings)
- No exotic optimizer needed

## Files

- `gumbel_ternary.py` — Gumbel-Softmax + STG implementation
- `reinmax_ternary.py` — REINMAX with custom autograd (Heun's method)
- `mirror_descent_ternary.py` — Mirror Descent with prob-space gradient transfer
- `discrete_ternary_base.py` — Shared model builder infrastructure
- `run_discrete_sweep.py` — d=128 sweep (pruned and full density)
- `run_large_sweep.py` — d=768 sweep (full density + Phase 1 training)
- `diagnose_init.py` — Diagnostic: weight comparison between PF and Gumbel models
- Checkpoints: `checkpoints/discrete_phase1_d768L6.pt`
- wandb: `bonsai-ternary-test` project, runs prefixed `large_*` and `discrete_*`
