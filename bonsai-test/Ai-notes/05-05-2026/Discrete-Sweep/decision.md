# Decision: STE+Adam for Ternary Training — 2026-05-05

## The Answer

**STE+Adam is the ternary training method.** No exotic discrete optimizer needed.

## What We Tested

6 methods across 2 scales (13M and 118M params):
- STE+Adam (straight-through estimator + Adam on continuous shadow weights)
- Gumbel-Softmax, Straight-Through Gumbel, REINMAX (NeurIPS 2023), Mirror Descent
- Pressure Flip, NES-Categorical, LR-nets (from overnight sweep)

## Why STE+Adam Wins

1. **Only method that actually changes ternary assignments.** Shadow weight updates cross quantization boundaries naturally. Logit-based methods (Gumbel/REINMAX) never cross argmax boundaries — literally zero learning at 118M scale.

2. **Ternary tax vanishes at scale.**
   - 13M params: 0.39 nats (48% PPL cost)
   - 118M params: 0.13 nats (14% PPL cost)
   - Extrapolated 1B+: <0.05 nats (~free)

3. **bf16 optimizer states are free.** Adam m1/m2 in bf16 = identical quality to fp32. 8 bytes/param total (2 shadow + 2 m1 + 2 m2 + 2 ternary).

## Recipe

```python
# Phase 1: Train from scratch with STE+Adam
model = TernaryTransformer(cfg)
optimizer = AdamW(model.parameters(), lr=3e-4)  # 3e-4 for 100M+, 6e-4 for <100M
# Standard training loop — STE quantization is built into BitLinear.forward()

# Phase 2 (optional): Prune with CMS to target density, continue STE+Adam
# The ternary tax is small enough that pruning is the bigger lever for efficiency
```

## What NOT to Do

- SGD for ternary (diverges catastrophically)
- Gumbel-Softmax / REINMAX / any stochastic discrete method (zero learning at scale)
- Pressure Flip without Adam-level adaptivity (plateaus at 5.96 vs STE's 5.33)
- Worry about the ternary tax at scale (0.13 nats at 118M, shrinking)
