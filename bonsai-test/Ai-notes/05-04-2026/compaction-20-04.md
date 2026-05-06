# Context Dump — 2026-05-04 20:04

## Current Work

Building NES-Categorical trainer for ternary {-1,0,1} neural networks. This is a PIVOT from EGGROLL (which was proven incompatible with ternary).

**Immediate next step:** Build `NESCategoricalTrainer` class, then run ablation experiments A4 → A1 → B1 → C2.

## Key Files
- `Ai-notes/05-04-2026/NES-Ternary-Experiment-Plan.md` — **THE MASTER PLAN. Read this first.**
- `kernels/eggroll_fast.py` — FastPopulationEvaluator (reusable: chunked CE, torch.compile)
- `run_ab_eggroll.py` — A/B test harness (add NES-Cat as new arm)
- `eggroll_model.py` — EggrollTransformer (basis for weight structure)
- `checkpoints/ab_phase1_d128L3.pt` — Phase 1 checkpoint (d=128, 3L, STE-trained 5k steps)
- `kernels/eggroll_fwd.py` — OLD broken Triton kernel, DO NOT USE
- `kernels/bench_eggroll_kernel.py` — benchmarks for fast evaluator
- `kernels/test_eggroll_fast.py` — correctness tests

## Footguns & Gotchas

1. **EGGROLL CANNOT work for ternary** — ±1 step is the ENTIRE weight range. Proven across all hyperparameter settings. Don't retry.
2. **Skip embed/lm_head in ES** — they must stay intact (not pruned, not perturbed). The d=128 model has 6.3M embed + 6.3M lm_head vs only 118k linear params.
3. **SM_120 (5090)**: torch.compile mode='default' NOT 'max-autotune' (101KB shmem limit breaks autotuning). num_stages=1, num_warps=4 for Triton.
4. **Zombie GPU processes** — check nvidia-smi before starting. Previous sessions left zombies that eat 5+ GB VRAM.
5. **Data loader** — creates new OpenWebText stream each call. eval_model creates separate stream. Training and eval see different data.
6. **Two zombie processes were killed** this session (PIDs 3165817, 3167913) from a session 3 hours old.

## Decisions Made

- **EGGROLL → NES-Categorical pivot**: EGGROLL's ±1 update is 0.4% for int8 but 100% for ternary. The convergence theorem requires quasi-continuous space. Ternary has 3 points per dimension — too discrete.
- **Block-coordinate cycling**: Instead of pop=118k (full space), partition into 21 matrices of ~5.6k each, pop=256 per matrix. Momentum carries signal. 43× faster per cycle.
- **2-DOF logit parameterization**: θ_neg, θ_pos (θ_zero=0). 4 bytes/param vs AdamW's 13 bytes.
- **CDF-inversion antithetical pairs**: u and 1-u for categorical. Valid variance reduction.
- **d=128 model for ablations**: 118k trainable linear params at 20% density. Full d=512 model (19.8M params) needs multi-GPU for NES.

## Incomplete / Next Steps

- [ ] Build `NESCategoricalTrainer` class with cycling + momentum
- [ ] Build `LRNetTrainer` class (CLT reparameterization, backprop on distributions)
- [ ] Run A4 (per-matrix cycling + momentum) — main hypothesis
- [ ] Run A1 (full-space NES) — control
- [ ] Run B1 (LR-nets) — theoretical gold standard
- [ ] Run C2 (STE→NES warm start) — pipeline dream

## Context That Won't Survive

- The fast evaluator (kernels/eggroll_fast.py) gives 3.2× speedup via: skip embed/lm_head perturbation, chunked CE, torch.compile. Benchmark data in Ai-notes/05-04-2026/EGGROLL-Kernel-Results.md.
- Three opus research subagents ran in parallel: math derivation, literature search, system design. All findings consolidated in the experiment plan.
- The LR-nets paper (ICLR 2018, arXiv:1710.07739) is the closest prior work — they maintain per-weight probability distributions over ternary, but use CLT reparameterization instead of REINFORCE because score-function estimator has too high variance.
- The math agent proved: natural gradient for categorical NES is just the "advantage" E[L|w=k] - E[L]. Fisher matrix is FREE from softmax. No Hessian computation needed.
- Population requirement: N = Ω(P/Δ²) for full space, where Δ is marginal advantage gap. With G groups: N_per_group = Ω(P/(G·Δ²)). This is why cycling helps.
- IndeCateR (NeurIPS 2023) achieves low variance with only 2 samples for categorical — potential variance reduction technique.

## Baseline Results (Already Collected)

| Setting | val_loss | val_ppl | Time |
|---------|----------|---------|------|
| STE+Adam, 80% density, d=128 | 5.36 | 212.6 | 45s |
| STE+Adam, 20% density, d=128 | 5.48 | 239.2 | 46s |
| EGGROLL, 20% density, d=128, pop=2048 | ~9.1 (plateau) | ~9000 | diverged |
| EGGROLL, 20% density, intact embed/lm_head | ~6.7 (oscillating) | ~800 | oscillating |

## VLT Thread
- `bonsai-ternary` — pushed NES-Categorical pivot summary
