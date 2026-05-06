# Context Dump — 2026-05-05 03:15

## Current Work

Building native ternary {-1,0,1} training methods that don't require continuous shadow weights. Started from EGGROLL (proven incompatible), pivoted through NES-Categorical, LR-nets, and Pressure Flip. Now exploring differentiable discrete optimization (Gumbel-Softmax family).

## Complete Scoreboard (d=128, L=3, 20k Phase 2 steps, 20% density)

| Method | Val Loss | Val PPL | Mem/param | Speed | Status |
|--------|----------|---------|-----------|-------|--------|
| bf16 full (no ternary) | 4.94 | 140 | 14 bytes | 131 step/s | Ceiling |
| STE+Adam bf16 states | **5.33** | **207** | **8 bytes** | ~120 step/s | **Best practical** |
| STE+Adam fp32 | 5.33 | 206 | 14 bytes | 120 step/s | Standard baseline |
| Hybrid STE 2k→PF 18k | 5.73 | 309 | varies | ~80 step/s | Best PF |
| NES-Cat A4 (cycling+mom) | 5.84 | 343 | 8 bytes | 3.2 step/s | Gradient-free |
| PF Annealed 1000→50 | 5.87 | 353 | 3 bytes | ~80 step/s | — |
| LRNet B1 (CLT reparam) | 5.93 | 375 | ~10 bytes | 76 step/s | — |
| PF Top-K (any K) | 5.96 | 388 | 3 bytes | ~80 step/s | All K same! |
| PF bf16 low-thresh | 6.06 | 428 | 3 bytes | ~90 step/s | — |
| STE+SGD (no Adam) | 10.65 | 42081 | 6 bytes | ~130 step/s | DIVERGED |
| EGGROLL | ~9.1 | ~9000 | 1 byte | — | INCOMPATIBLE |

## Key Findings

### 1. Adam is NON-NEGOTIABLE for ternary
STE+SGD completely diverges. The per-weight variance normalization (m1/sqrt(m2)) is essential. Any method without per-weight adaptivity caps at ~5.96.

### 2. bf16 optimizer is free
STE+Adam with bf16 m1/m2 = identical to fp32. Immediate 43% memory savings (14→8 bytes).

### 3. Flip rate doesn't matter
Top-K sweep at 50/200/500/1000 all give 5.96. The PF plateau is from lacking per-weight adaptivity, not wrong flip count.

### 4. The gap decomposition
- Ternary tax: 0.39 nats (bf16→STE, unavoidable at this model size)
- Adaptivity tax: 0.40 nats (STE→PF, needs per-weight LR)
- Flip rate: ~0 nats (proven irrelevant)

### 5. Pressure Flip's initial results were misleading
At 5k steps PF matched STE (5.50 vs 5.48). At 20k steps PF plateaus (5.96) while STE keeps improving (5.33). The "easy flips" happen fast but the remaining optimization requires continuous adaptation that int8/bf16 counters can't provide.

## Footguns & Gotchas

1. **STE+SGD diverges** — never use SGD for ternary, always Adam
2. **NES-Cat initial logits**: use θ=±2, NOT ±10 (gradient vanishes at ±10, p_peak=99.3%)
3. **PF threshold schedule**: cosine ramp kills learning at long horizons — constant or Top-K is better
4. **Phase 1 checkpoints**: `ab_phase1_d128L3.pt` (B=12, 5k steps) vs `bench_phase1_d128L3.pt` (B=8, 5k steps) — different starting points, slightly different results
5. **Module name collision in pressure_flip.py**: `_build_pf` pops from pruned_data dict by shape match — same-shaped layers can collide. Works for d=128 (unique shapes) but needs fixing for larger models
6. **SM_120**: num_stages=1, num_warps=4, mode='default' not 'max-autotune'

## Next Steps — Differentiable Discrete Optimization

Tasks #103-108. Priority order:

1. **Gumbel-Softmax** (#103) — Standard tool for discrete+backprop. Per-weight logits, Gumbel noise, temperature anneal. This is how NAS/VQ-VAE solve the same problem. Most likely to work.

2. **REINMAX** (#104) — NeurIPS 2023, improved STE for categorical. Lower bias+variance than Gumbel. Paper: arXiv:2304.08612.

3. **Straight-Through Gumbel** (#105) — Hard forward (exact ternary) + soft backward (Gumbel gradient). Best of both worlds.

4. **Mirror Descent** (#106) — Natural geometry for categorical is KL on simplex. Information-geometric approach.

5. **Min-distortion projection** (#107) — Better projection than STE's rounding.

6. **Scale test** (#108) — Run best method at d=512. The ternary tax likely shrinks with scale.

## Key Files

- `run_ab_eggroll.py` — unified test harness (arms: baseline, eggroll, nes_cat, lrnet, pressure_flip)
- `nes_categorical.py` — NES-Categorical trainer (batched forward, cycling, momentum)
- `lrnet.py` — LR-nets CLT reparameterization
- `pressure_flip.py` — PressureFlip (detach+requires_grad gradient capture)
- `overnight_pf.py` — 9-experiment overnight sweep
- `bench_pressure_flip.py` — 3-way bf16/STE/PF benchmark
- `eggroll_model.py` — EggrollTransformer (explicit ternary buffers)
- `model.py` — TernaryTransformer (STE shadow weights)
- `bitlinear.py` — BitLinear, STE, ternary_quantize
- `kernels/eggroll_fast.py` — FastPopulationEvaluator + chunked CE

### Checkpoints
- `checkpoints/ab_phase1_d128L3.pt` — Phase 1 d=128 (B=12, 5k steps)
- `checkpoints/bench_phase1_d128L3.pt` — Phase 1 d=128 (B=8, 5k steps, used for overnight)
- `checkpoints/pipeline_v3_phase1.pt` — Phase 1 d=512 (may work for scale test)

### Results & Notes
- `Ai-notes/05-05-2026/PressureFlip-Overnight/results.md` — full overnight analysis
- `Ai-notes/05-04-2026/NES-Ternary-Experiment-Plan.md` — original experiment plan
- `Ai-notes/05-04-2026/EGGROLL-Kernel-Results.md` — kernel benchmarks
- wandb: `bonsai-ternary-test` project (all runs logged)

## Context That Won't Survive

- The research direction was Wolfe's insight: "we need shadow weights without materializing all of them" → "what state to be in, not what direction to move" → led to Pressure Flip
- Wolfe suspects there's "a solution to this problem" — native ternary training. Not ready to give up and just train bf16. The motivation is memory efficiency for inference (ternary = 2 bits/weight).
- The conversation arc: EGGROLL kernel → EGGROLL fails → NES-Categorical pivot → LR-nets comparison → Pressure Flip invention → overnight sweep → gap analysis → Gumbel-Softmax direction
- VLT thread: `bonsai-ternary` (pushed NES-Cat pivot summary earlier)
- The user prefers to discuss ideas conceptually before implementation. "Unleash the subagents" means parallel research. "Build it" means write code.
