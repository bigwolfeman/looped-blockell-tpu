# Neural Memory Performance Optimization Plan

## Current State (2026-05-01)
- **mem_residual** running: 100k steps, val_ppl 35.6 @ step 8500
- Speed: 1.7-1.8 step/s (baseline: 3.4 step/s = 47% overhead)
- VRAM: 20.3GB (baseline: 15-16GB)
- Memory: 6-layer MLP, d_memory=1024, K=5 inner steps, theta=0.01
- Residual mode: `h_new = h_new + scale * norm(proj(mem_out))`

## Bottleneck Analysis
The 47% overhead comes from K=5 inner gradient steps:
- Each step: forward MLP (~0.5ms) + autograd.grad (~1ms) + weight update (~0.1ms)
- At full batch B=20, S=1024: 5 × ~1.6ms = ~8ms per training step
- But autograd.grad overhead scales with graph complexity, actual ~20-30ms

## Optimization Targets

### 1. Reduce K (Quick Win)
- Test K=3 and K=2 on short runs (2-3k steps)
- Compare memory_loss convergence and task_loss impact
- K=2 would save 60% of inner loop time

### 2. Fused Triton Kernel (Big Win)
- `titans_core/kernels/memory_mlp.py` has a template
- Fuse: forward + grad_compute + grad_clip + momentum_update + weight_update
- Single kernel launch vs 5 × (forward + autograd.grad + python update)
- Expected: 3-5× speedup on inner loop

### 3. Chunk Inner Loop
- Instead of K=5 on full [B, S, d], do K=5 on [B, S//K, d] chunks
- Same total gradient signal, 1/K memory per inner step
- Reduces peak VRAM from 20.3GB toward baseline

## Experiment Plan (post-ablation)
1. K=2 short run (2k steps) — compare mem_loss and task_loss vs K=5
2. K=3 short run (2k steps) — same comparison
3. Pick best K, run 5k steps to verify stability
4. Build fused kernel if K>1 still needed
5. Final 15k comparison run at optimized settings

## Key Files
- `interop/pt_model.py` — model with memory integration
- `titans_core/memory/neural_memory.py` — NeuralMemory class
- `titans_core/kernels/memory_mlp.py` — existing fused kernel template
- `scripts/run_ablation_pt.py` — training script
- `configs/ablation_100k.yaml` — 100k config

## What Worked / What Failed
- **Logit bias mode**: FAILED — modifying Q in attention is multiplicatively unstable
- **Residual mode**: STABLE — additive integration, zero-init scale
- **SIGReg**: USELESS — constant log(2), zero gradient, 2× speed cost
- **Memory warmup**: CRITICAL — 1k backbone-only + 4k ramp prevents destabilization
- **Detach memory_loss**: REQUIRED — prevents outer optimizer double-updating MLP
- **Alpha scaling**: Original paper values (0.03) too aggressive for per-step updates
