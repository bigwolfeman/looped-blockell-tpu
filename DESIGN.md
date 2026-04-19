# Looped Block-ELL Transformer — TPU Implementation

## Architecture

Parcae-style looped transformer with Block-ELL sparse MLPs, targeting TPU (Colab v2/v5e + Trillium v6e).

**Core idea**: 6 unique transformer layers (1 prelude + 4 core + 1 coda), core looped T×8 with diagonal injection for stability. Dense → prune → compact → route.

### Two-Level Tile Hierarchy

| Level | Size | Purpose |
|-------|------|---------|
| Tile | 16×16 | CMS scoring, pruning decisions |
| Macro-block | 128×128 | TPU MXU dispatch (8×8 tiles) |

### Column Reorder Compaction

d_ff is internal to MLP (fc1 output = fc2 input). At each prune round:

1. Score each d_ff feature by alive-tile count across fc1+fc2
2. Sort d_ff by importance → column permutation
3. Apply permutation: fc1 cols, fc2 rows, optimizer state
4. K_active_macros = number of non-empty macro-columns
5. Pallas kernel iterates only K_active_macros → instant speedup

At final compaction: physically truncate d_ff, rebuild state, one XLA recompile.

### Three-Phase Pipeline

```
Phase B.1 (Dense):    Steps 0→prune_start    — Block-ELL at density=1.0, CMS scoring
Phase B.2 (Prune):    prune_start→compact    — Gradual pruning + column reorder each round
Phase C   (Route):    compact→end            — Compact + ReMoE routing + iteration embedding
```

## Framework

- **JAX + Flax linen** (not PyTorch/XLA — no sparse tensor support)
- **Pallas** for Block-ELL kernels (scalar prefetch for tile indices)
- **optax** for optimization
- **lax.scan** for weight-sharing loop (constant memory, no unrolling)
- **jax.checkpoint** on scan body for activation checkpointing

## Module Map

```
looped_blockell/
├── __init__.py
├── config.py              # Dataclass config (no neural memory fields)
├── layers/
│   ├── __init__.py
│   ├── block_ell.py       # Block-ELL tensor format (JAX arrays)
│   ├── block_linear.py    # CMSBlockLinear equivalent (Flax module)
│   ├── mlp.py             # MLP block (dense or block-sparse)
│   ├── attention.py       # Multi-head attention with RoPE
│   └── norms.py           # RMSNorm
├── kernels/
│   ├── __init__.py
│   └── block_ell_matmul.py  # Pallas kernel with macro-block dispatch
├── looping/
│   ├── __init__.py
│   ├── looped_model.py    # LoopedTransformer (lax.scan core)
│   ├── diagonal_injection.py  # SSM-style h = decay*h + dt*e
│   └── depth_sampler.py   # Poisson depth + truncated BPTT
├── opt/
│   ├── __init__.py
│   ├── cms.py             # CMS scoring scheduler
│   ├── topology.py        # Topology scorer + decisions
│   └── column_reorder.py  # d_ff permutation for macro-block density
├── routing/
│   ├── __init__.py
│   ├── remoe_router.py    # ReMoE (ReLU gates + adaptive L1)
│   └── routed_mlp.py      # RoutedMLP wrapper
scripts/
├── train.py               # Full pipeline training script
notebooks/
├── looped_blockell_colab.ipynb  # Self-contained Colab notebook
tests/
├── test_block_ell.py
├── test_looped_model.py
├── test_column_reorder.py
└── test_pallas_kernel.py
configs/
├── small.yaml             # d=768, 1+4+1, for Colab
├── medium.yaml            # d=1536, 3+6+3, for single Trillium
└── large.yaml             # d=2048+, 3+8+3, for 3-node Trillium
```

## Key Design Decisions

1. **lax.scan for looping**: Fixed max iterations, jnp.where for per-sequence freeze
2. **No Triton**: All custom kernels via Pallas (TPU-native)
3. **Macro-block skip via loop bound**: K_active_macros controls compute, not masking
4. **Column reorder at each prune round**: Clusters alive tiles, no shape change needed
5. **bf16 throughout**: MXU optimized for bfloat16
6. **Multi-host FSDP**: jax.sharding for 3-node training, drop to 1 after compact
