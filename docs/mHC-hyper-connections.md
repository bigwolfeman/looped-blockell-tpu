# mHC — Manifold-Constrained Hyper-Connections

**Paper**: arXiv:2512.24880 (Dec 2025, DeepSeek)
**Hyperloop**: arXiv:2604.21254 (Apr 23 2026, MIT — applies mHC to looped transformers)

## What It Does

Generalizes residual connections. Instead of `x_{l+1} = x_l + F(x_l)`, expands the
residual stream into **n parallel streams** (n=4 typical) with learned mixing:

```
x_{l+1} = H_l^res · x_l + H_l^post^T · F(H_l^pre · x_l, W_l)
```

- `H_l^res` (n×n): mixes features within the residual stream — **most important**
- `H_l^pre` (1×n): aggregates nC-dim stream to C-dim for layer input
- `H_l^post` (1×n): projects layer output back into stream

## Stability Constraint

`H_l^res` is constrained to be **doubly stochastic** (Birkhoff polytope) via
**Sinkhorn-Knopp** (20 iterations of alternating row/col normalization):
- Spectral norm bounded by 1 → cannot amplify signals
- Closed under composition → depth doesn't degrade stability
- Without constraint: loss surges at ~12k steps, gain magnitude spikes to 3000x
- With mHC: gain stays ~1.6x

## Evolution of the Idea

```
HC (ByteDance, unconstrained) 
  → mHC (DeepSeek, doubly stochastic via Sinkhorn)
  → JPmHC (JP Morgan arXiv:2602.18308, orthogonal via Cayley transform)
  → Hyperloop (MIT arXiv:2604.21254, diagonal parameterization for loops)
```

JPmHC argues Sinkhorn/doubly-stochastic is flawed (eigenvalues contract inside unit
disk). Proposes Cayley-transform orthogonal constraints (Stiefel manifold) — 2.25x
fewer FLOPs, better gradient flow.

## Hyperloop: mHC + Looped Transformers

Architecture: 2 begin + 4 middle (looped 3×) + 2 end, HC at loop boundaries only.

| Model | Params | PPL |
|-------|--------|-----|
| Standard Transformer | 2018M | 8.60 |
| mHC Transformer | 2033M | 8.57 |
| Vanilla Looped | 990M | 8.68 |
| **Hyperloop** | **991M** | **8.49** |

Hyperloop beats 2B transformer with **51% fewer parameters**.

### Key Finding: HC Prevents Representation Collapse
Without HC, looped layers converge to identical outputs across iterations.
With HC, representations diverge productively (cosine sim drops 0.7429 → 0.7382).

### Ablations
- Removing HC from looped model: +0.45 PPL
- Sinkhorn (mHC-style) vs diagonal parameterization: **diagonal is better for loops** (+0.19 PPL penalty for Sinkhorn)
- HC every layer vs per-loop-only: per-loop slightly better
- Overhead: only 150-300K params (~0.4% for 81M model)

## Relevance to Our Architecture

1. **Our diagonal injection IS a form of hyper-connection** — Parcae's `h = decay*h + dt*e`
   is a single-stream diagonal HC. mHC expands to n streams.

2. **Loop-boundary HC solves representation collapse** — our Poisson depth + jnp.where
   freeze partially addresses this, but HC is more principled.

3. **Orthogonal to Block-ELL** — HC operates on residual stream (d_model), Block-ELL
   on MLP weights (d_ff). They're complementary. Hyperloop doesn't explore sparse
   MLPs at all — **unexplored territory**.

4. **Ablation**: Compare our Parcae diagonal injection vs loop-boundary HC (diagonal
   parameterization per Hyperloop). The diagonal version is trivial: n learnable
   scalars per loop boundary.

5. **Outer SSM interaction**: Our outer SSM loop is effectively a second HC at the
   sequence boundary. mHC + outer SSM = hierarchical hyper-connections (inner loop
   boundaries + outer sequence boundaries).

## Implementation Notes

- Diagonal parameterization (Hyperloop) for loops: `n` learnable scalars per boundary
- Apply at loop boundaries only, not every layer
- 6.7% training overhead with n=4 (kernel fusion needed for full speed)
- GitHub: https://github.com/Kareem404/hyper-connections
