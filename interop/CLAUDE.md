# CLAUDE.md — Interop (PyTorch ↔ JAX)

## Purpose

Bidirectional checkpoint conversion between PyTorch and JAX/Flax for the
Looped Block-ELL Transformer. Enables fast ablation on GPU (PyTorch) with
seamless continuation on TPU (JAX).

## Key Files

- `pt_model.py` — PyTorch model that exactly mirrors the JAX architecture
- `convert_checkpoint.py` — Bidirectional converter with CLI

## Critical: Block-ELL Tile Convention

The JAX einsum `"bsrkd,rkdD->bsrD"` treats:
- d = input sub-index (contracts with x_gathered)
- D = output sub-index

So values must be `[R, K, B_in, B_out]`. When converting from PyTorch
`nn.Linear` weight `[out, in]`:
```
weight.reshape(R, B, C, B).transpose(0, 2, 3, 1)  → [R, C, B_in, B_out]
```
NOT `.transpose(0, 2, 1, 3)` which gives `[R, C, B_out, B_in]`.

## Name Mapping

| PyTorch | JAX/Flax |
|---------|----------|
| `embed.weight` | `params/embed/embedding` |
| `prelude.0.X.weight` | `params/prelude_0/X/kernel` |
| `core.N.mlp.fc1.weight` | `params/core_N/mlp/fc1/values` (Block-ELL) |
| `injection.log_A` | `params/injection/log_A` |
| `input_norm.scale` | `params/input_norm/scale` |

## Tests

`pytest tests/test_interop.py` — 24 tests covering name mapping, Block-ELL
round-trip, full checkpoint conversion, optimizer state, model forward/backward.
