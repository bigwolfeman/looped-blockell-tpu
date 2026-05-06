"""Diagnose why discrete ternary models start at val=8.77 instead of ~6.5.

Compare effective weights between PressureFlip (known-good) and Gumbel (broken init).
"""
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

device = torch.device("cuda")

# --- Load + prune (first copy for PF) ---
from model import TernaryTransformer
from run_ab_eggroll import perlayer_prune
from bitlinear import BitLinear, BitEmbedding

ckpt = torch.load("checkpoints/bench_phase1_d128L3.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
model1 = TernaryTransformer(cfg).to(device)
model1.load_state_dict(ckpt["model_state_dict"])
pd1, _ = perlayer_prune(model1, 0.20, device, skip_embed_lmhead=True)

from pressure_flip import create_pressure_flip_model
pf_model = create_pressure_flip_model(model1, pd1, cfg, device)

# --- Load + prune (second copy for Gumbel) ---
model2 = TernaryTransformer(cfg).to(device)
model2.load_state_dict(ckpt["model_state_dict"])
pd2, _ = perlayer_prune(model2, 0.20, device, skip_embed_lmhead=True)

from gumbel_ternary import GumbelTernaryLinear
from discrete_ternary_base import create_discrete_model
gumbel_model = create_discrete_model(model2, pd2, cfg, device, GumbelTernaryLinear)

# --- Compare effective weights ---
print("\n=== Weight comparison ===")
pf_mods = list(pf_model.pf_modules)
gumbel_mods = list(gumbel_model.discrete_modules)

gumbel_model.eval()
pf_model.eval()

for i in range(min(3, len(pf_mods))):
    pf_name, pf_mod = pf_mods[i]
    gm_name, gm_mod = gumbel_mods[i]

    # PF effective weight
    w_pf = pf_mod._effective_weight()

    # Gumbel effective weight at eval (near-hard)
    from discrete_ternary_base import build_full_logits, apply_scales_and_mask
    full = build_full_logits(gm_mod.logits, device)
    y = F.softmax(full / 0.01, dim=-1)
    w_soft = y[..., 2] - y[..., 0]
    w_gumbel = apply_scales_and_mask(
        w_soft, gm_mod.scales, gm_mod.mask,
        gm_mod.out_features, gm_mod.in_features, gm_mod.group_size)

    diff = (w_pf - w_gumbel).abs()
    print(f"\n  {pf_name} vs {gm_name}:")
    print(f"    PF weight:     mean={w_pf.abs().mean():.6f}  min={w_pf.min():.4f}  max={w_pf.max():.4f}")
    print(f"    Gumbel weight: mean={w_gumbel.abs().mean():.6f}  min={w_gumbel.min():.4f}  max={w_gumbel.max():.4f}")
    print(f"    Max diff: {diff.max():.6f}  Mean diff: {diff.mean():.6f}")
    print(f"    PF shape: {w_pf.shape}  Gumbel shape: {w_gumbel.shape}")

    # Check ternary recovery
    w_ternary = w_soft.round()
    t_pf = pf_mod.ternary.float()
    ternary_match = (w_ternary == t_pf).float().mean()
    print(f"    Ternary match: {ternary_match:.4f}")

# --- Compare full model eval ---
print("\n=== Evaluation comparison ===")
from run_ab_eggroll import get_data_loader
loader = get_data_loader(batch_size=8, seq_len=256)
n_batches = 5

# Phase 1 (raw, after pruning)
model2_eval = TernaryTransformer(cfg).to(device)
model2_eval.load_state_dict(ckpt["model_state_dict"])
model2_eval.eval()
losses_raw = []
eval_data = []
for i, (x, y) in enumerate(loader):
    if i >= n_batches:
        break
    x, y = x.to(device), y.to(device)
    eval_data.append((x, y))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model2_eval(x, labels=y)
    losses_raw.append(out["loss"].item())
print(f"  Phase 1 (unpruned): val={sum(losses_raw)/len(losses_raw):.4f}")

# PF model
losses_pf = []
for x, y in eval_data:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = pf_model(x, labels=y)
    losses_pf.append(out["loss"].item())
print(f"  PressureFlip init:  val={sum(losses_pf)/len(losses_pf):.4f}")

# Gumbel model
losses_gm = []
for x, y in eval_data:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = gumbel_model(x, labels=y)
    losses_gm.append(out["loss"].item())
print(f"  Gumbel init:        val={sum(losses_gm)/len(losses_gm):.4f}")

# Check: does Gumbel with TRAIN mode (soft weights) differ?
gumbel_model.train()
losses_gm_train = []
for x, y in eval_data:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = gumbel_model(x, labels=y)
    losses_gm_train.append(out["loss"].item())
print(f"  Gumbel TRAIN mode:  val={sum(losses_gm_train)/len(losses_gm_train):.4f}")

# Check embed identity
print(f"\n  Embed same object? {pf_model.embed is model1.embed}")
print(f"  LM head same object? {pf_model.lm_head is model1.lm_head}")
print(f"  Gumbel embed from model2: {gumbel_model.embed is model2.embed}")
