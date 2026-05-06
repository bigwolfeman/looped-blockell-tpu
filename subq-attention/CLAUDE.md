# CLAUDE.md — SubQ Attention Ablation

## Goal

Reverse-engineer subquadratic sparse attention (inspired by SubQ/SSA claims).
Ablate candidate architectures to find the best subquadratic selection mechanism.

## Architecture

Pluggable attention transformer: d=768, 12 heads, 6 layers, ~117M params.
Attention modules registered in `attention.py` via `@register_attention("name")`.

## Files

- `model.py` — SubQTransformer with pluggable attention
- `attention.py` — Attention registry + base implementations (dense, sliding_window, nsa, mosa, nsa_mosa, etc.)
- `attention_cosine.py` — NSA+Cosine similarity router (MegaContext internalized)
- `attention_optimized.py` — GQA, QK-Norm, XSA, Residual Attention, CoPE, combined
- `attention_mla.py` — Multi-head Latent Attention (DeepSeek V2 simplified)
- `attention_cla.py` — Cross-Layer Attention, per-layer Mixing Gate
- `train.py` — Training loop with wandb, --attn_kwargs JSON support, OpenWebText streaming
- `run_optimizations.sh` — Ablation launcher for optimization experiments

## Key Patterns

- Every attention module inherits `BaseAttention` (shared Q/K/V projections + RoPE)
- `forward(x: [B,S,D]) -> [B,S,D]` — causal masking handled internally
- `create_attention(cfg, layer_idx)` — factory from registry
- `cfg.attn_kwargs` passes method-specific params (window_size, n_buckets, etc.)
- Weight-tied embeddings (lm_head.weight = embed.weight)

## Commands

```bash
python train.py --attn dense --name dense_4k --seq_len 4096 --steps 20000
python train.py --attn gqa --name gqa_test --steps 12000 --attn_kwargs '{"n_kv_heads": 4}'
python train.py --attn mla --name mla_test --steps 12000 --attn_kwargs '{"kv_rank": 256}'
python train.py --attn dense_cope --name cope_test --attn_kwargs '{"context_len": 2048}'
bash run_optimizations.sh  # runs all 8 optimization ablations
```

## Experiments

### Phase 1: Subquadratic Selection (complete)

| Exp | Attention Type | Selection | Complexity |
|-----|---------------|-----------|------------|
| 0 | dense | Full Q·K | O(n²) |
| 1 | sliding_window | Position-based | O(n·w) |
| 2 | nsa | Quadratic top-k | O(n²) select |
| 3 | nsa_lsh | LSH hash buckets | O(n·d) |
| 5 | nsa_block_router | Learned block scoring | O(n/B·d) |
| 6 | lod_attention | Hierarchical tree search | O(n·log n) |
| 7 | mosa | MoSA per-head routing | O(k²·d) |
| 8 | nsa_mosa | NSA + MoSA router | O(n²/r + nk) |
| 9 | nsa_cosine | NSA + cosine sim | O(n²/r + nk) |

### Phase 2: Attention Optimizations (running)

| Type | Optimization | Key Param |
|------|-------------|-----------|
| gqa | Grouped KV heads | n_kv_heads=4 |
| qk_norm | RMSNorm on Q,K | — |
| xsa | Exclude self-token | — |
| residual_attn | Learned residual α·x | init_alpha=0.1 |
| mla | Low-rank KV bottleneck | kv_rank=256 |
| dense_cope | Clipped RoPE taper | context_len=seq_len |
| cla | Cross-layer KV reuse | share_interval=2 |
| mixing_gate | Dense↔MoSA blend | tokens_per_head=512 |
| dense_opt | GQA+QK-Norm+Residual+CoPE | combined |

## Wandb

Project: `subq-attention-ablation`

## Phase 1 Results (Final)

| Method | bs | Val PPL @ 4k steps | vs Dense | Complexity |
|---|---|---|---|---|
| Dense | 8 | 54.0 | ceiling | O(n²) |
| Dense | 4 | 69.9 | token-matched | O(n²) |
| NSA | 4 | 75.5 | +8% vs dense bs=4 | O(n²/r) |
| NSA+MoSA | 8 | 52.2 | -3.3% vs dense bs=8 | O(n²/r + nk) |
| Cosine | 4 | 70.5 | +0.9% vs dense bs=4 | O(n²/r + nk) |

MoSA routing is the winner. NSA+MoSA beats dense. Pure MoSA+window = O(n) linear.

## Critical

- 5090 does NOT support FlashAttention — SDPA falls back to math kernel
- bf16 autocast for activations, fp32 for optimizer states
- RoPE required for positional info in all attention variants
- CLA uses global dict cache — layers MUST execute in order (layer 0 clears cache)
- CoPE context_len should match training seq_len for correct frequency tapering
- XSA: first token has no valid keys (all -inf → zero attention output)
