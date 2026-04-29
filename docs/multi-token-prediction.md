# Multi-Token Prediction (MTP)

**Key papers**: Gloeckle et al. 2024 (arXiv:2404.19737), DeepSeek-V3/V4, MuToR (NeurIPS 2025, arXiv:2505.10518)
**JAX impl**: MaxText `MultiTokenPredictionBlock` in `layers/decoders.py`

## Two Approaches

### Meta (Gloeckle): Parallel Independent Heads
- n independent transformer layers on shared trunk output
- Each head predicts t+i independently
- n=4 optimal for code at 7B+
- Sequential backward trick: O(V+d) memory instead of O(nV+d)

### DeepSeek-V3: Sequential Chain (D=1)
- Single extra prediction depth
- `h' = M_k[RMSNorm(h); RMSNorm(Emb(t+k))]` — concat prev hidden + target embed
- λ=0.3 first 10T tokens → 0.1 for remaining 4.8T
- Gradients flow back to backbone

## Scale Threshold

| Scale | Effect |
|-------|--------|
| <1B params | **Hurts or neutral** — hidden state overcrowding |
| 1-3B | Mixed, task-dependent |
| 3B+ | Clear benefits on code |
| 7B+ | Strong benefits, n=4 optimal |

## For Our Architecture

**34M ablation model: DON'T USE MTP.** Literature is clear it hurts at this scale.

**700M+ TPU model: USE MTP with D=1, λ=0.1.** At effective ~4B (700M × 6 loops),
MTP should improve sample efficiency. DeepSeek uses D=1 even at 671B.

**Looped model interaction (unexplored):**
- Potential synergy: MTP gradients could accelerate fixed-point convergence
- Potential conflict: overcrowding from looping + MTP dual pressure on hidden state
- Start conservative: D=1, λ=0.05, monitor main loss

## MuToR Alternative (Better for Small Models)

Register tokens interleaved in input (~2K extra params). Showed benefits at 2B
fine-tuning. Outperformed standard MTP on GSM8K (42.1% vs 40.7% with Gemma 2B).
Better fit for small models since no separate head capacity needed.

## Implementation Notes

- Loss: `total = main_loss + λ * mean(mtp_losses)`, gradients flow to backbone
- Memory: sequential head computation (backward from farthest head first)
- DeepSeek decay: λ=0.3 early → 0.1 late
- MaxText default: `mtp_loss_scaling_factor=0.1`
