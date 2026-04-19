# CLAUDE.md — TPU Looped Block-ELL

## Overview

JAX/Flax port of the Looped Block-ELL Transformer for TPU scaling experiments.
Parent repo: TitanMAC-Standalone (PyTorch, GPU). This is a standalone JAX reimplementation.

## Architecture

Parcae-style looped transformer: prelude → core×T (with diagonal injection) → coda.
Block-ELL sparse MLPs with CMS gradient-based pruning. ReMoE routing post-compaction.

## Key Patterns

- `lax.scan` for the core loop (constant memory, no unrolling)
- `jax.checkpoint` on scan body for activation checkpointing
- Pallas kernels for Block-ELL matmul (scalar prefetch for tile indices)
- Two-level tiles: 16×16 pruning granularity, 128×128 macro-blocks for MXU
- Column reordering of d_ff at each prune round for macro-block density
- bf16 throughout, XLA full-graph compilation

## Commands

```bash
pip install -e .
pip install -e ".[dev]"
pytest tests/
python scripts/train.py --config configs/small.yaml
```

## Critical: CMS Score Timing

`accumulate_scores()` must be called with the gradients BEFORE the optax update is applied.
In JAX: grads = jax.grad(loss_fn)(params) → accumulate_scores(cms_state, grads) → params = optax.apply(params, grads)

## Critical: lax.scan + Poisson Depth

scan requires static iteration count. Use max_depth as the scan length.
Per-sequence freeze via jnp.where(active_mask, h_new, h). Frozen sequences waste FLOPs but produce correct gradients.

## Critical: Column Reorder

d_ff is internal to MLP — can be freely permuted. After each prune round:
1. Score d_ff features by alive-tile count
2. Sort by importance → permutation
3. Apply to fc1 columns, fc2 rows, optimizer state
4. K_active_macros = non-empty macro-columns → kernel loop bound
