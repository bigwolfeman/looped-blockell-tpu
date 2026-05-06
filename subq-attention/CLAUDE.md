# CLAUDE.md — SubQ Attention Ablation

## Goal

Reverse-engineer subquadratic sparse attention (inspired by SubQ/SSA claims).
Ablate candidate architectures to find the best subquadratic selection mechanism.

## Architecture

Pluggable attention transformer: d=768, 12 heads, 6 layers, ~117M params.
Attention modules registered in `attention.py` via `@register_attention("name")`.

## Files

- `model.py` — SubQTransformer with pluggable attention
- `attention.py` — Attention registry + implementations (dense, sliding_window, nsa, lsh, etc.)
- `train.py` — Training loop with wandb, scaling measurement, OpenWebText streaming

## Key Patterns

- Every attention module inherits `BaseAttention` (shared Q/K/V projections + RoPE)
- `forward(x: [B,S,D]) -> [B,S,D]` — causal masking handled internally
- `create_attention(cfg, layer_idx)` — factory from registry
- `cfg.attn_kwargs` passes method-specific params (window_size, n_buckets, etc.)
- Weight-tied embeddings (lm_head.weight = embed.weight)

## Commands

```bash
python train.py --attn dense --name dense_4k --seq_len 4096 --steps 20000
python train.py --attn sliding_window --name sw_4k --seq_len 4096 --window_size 4096
python train.py --attn dense --name dense_scaling --measure_scaling
```

## Experiments

| Exp | Attention Type | Selection | Complexity |
|-----|---------------|-----------|------------|
| 0 | dense | Full Q·K | O(n²) |
| 1 | sliding_window | Position-based | O(n·w) |
| 2 | nsa | Quadratic top-k | O(n²) select |
| 3 | nsa_lsh | LSH hash buckets | O(n·d) |
| 4 | nsa_product_key | PEER factored lookup | O(n·√k) |
| 5 | nsa_block_router | Learned block scoring | O(n/B·d) |
| 6 | lod_attention | Hierarchical tree search | O(n·log n) |

## Wandb

Project: `subq-attention-ablation`

## Critical

- 5090 does NOT support FlashAttention — SDPA falls back to math kernel
- bf16 autocast for activations, fp32 for optimizer states
- RoPE required for positional info in all attention variants
