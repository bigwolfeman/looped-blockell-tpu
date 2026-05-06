"""Overnight Pressure Flip experiments — 6 variants in sequence.

All share: d=128, L=3, B=8, seq=256, 20k Phase 2 steps, 20% density.
Phase 1 checkpoint: checkpoints/bench_phase1_d128L3.pt

Experiments:
  1. PF Top-K (fixed 200 flips/step)
  2. PF Annealed flip-rate (1000→50 cosine)
  3. PF bf16 momentum low-threshold (FLIP_MULT=0.05, no cooldown)
  4. Hybrid STE 2k → PF 18k
  5. STE + SGD momentum (no Adam) — 6 bytes/param
  6. STE + Adam bf16 optimizer states — 8 bytes/param

Usage:
  python overnight_pf.py           # run all
  python overnight_pf.py --exp 1   # run just experiment 1
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import wandb

torch.set_float32_matmul_precision("high")

WANDB_PROJECT = "bonsai-ternary-test"
B = 8
SEQ = 256
D_MODEL = 128
N_LAYERS = 3
VOCAB = 49152
PHASE2_STEPS = 20000
LR = 6e-4
LR_MIN = 6e-5
WARMUP = 500
DENSITY = 0.20
PHASE1_CKPT = "checkpoints/bench_phase1_d128L3.pt"


def get_loader():
    from run_ab_eggroll import get_data_loader
    return get_data_loader(batch_size=B, seq_len=SEQ)


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, device, n_batches=20):
    model.eval()
    loader = get_loader()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        losses.append(out["loss"].item())
    return sum(losses) / len(losses), math.exp(min(sum(losses) / len(losses), 20))


def load_phase1(device):
    from model import TernaryTransformer
    ckpt = torch.load(PHASE1_CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TernaryTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def prune_model(model, device):
    from run_ab_eggroll import perlayer_prune
    return perlayer_prune(model, DENSITY, device, skip_embed_lmhead=True)


# ===========================================================================
# Experiment 1: PF Top-K (fixed flip budget per step)
# ===========================================================================

def run_exp1(device, K=200):
    """Top-K flips: flip the top K weights by |pressure| every step."""
    from pressure_flip import create_pressure_flip_model
    model, cfg = load_phase1(device)
    pruned_data, density = prune_model(model, device)
    pf_model = create_pressure_flip_model(model, pruned_data, cfg, device)
    del model; torch.cuda.empty_cache()

    BETA = 0.9

    wandb.init(project=WANDB_PROJECT, name=f"overnight_pf_topk{K}",
               settings=wandb.Settings(init_timeout=180),
               config={"exp": 1, "method": "pf_topk", "K": K, "beta": BETA})

    # bf16 momentum pressure
    pressures = {}
    for name, mod in pf_model.pf_modules:
        pressures[name] = torch.zeros(mod.out_features, mod.in_features,
                                      dtype=torch.bfloat16, device=device)

    val0, ppl0 = evaluate(pf_model, device)
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)
    print(f"  [exp1] start: val={val0:.4f} ppl={ppl0:.1f}")

    loader = get_loader()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break
        x, y = x.to(device), y.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pf_model(x, labels=y)
        out["loss"].backward()
        running_loss += out["loss"].item()

        # Accumulate momentum
        all_pressures = []
        all_mods = []
        for name, mod in pf_model.pf_modules:
            if mod._w_leaf is None or mod._w_leaf.grad is None:
                continue
            grad = mod._w_leaf.grad
            if mod.weight_mask is not None:
                grad = grad * mod.weight_mask
            p = pressures[name]
            p.mul_(BETA).add_(grad.to(torch.bfloat16), alpha=-(1 - BETA))
            all_pressures.append(p)
            all_mods.append(mod)
            mod._w_leaf = None

        # Top-K selection across ALL modules
        flat_pressures = torch.cat([p.reshape(-1) for p in all_pressures])
        # Get masks for alive weights
        flat_alive = torch.cat([
            (mod.weight_mask.reshape(-1).bool() if mod.weight_mask is not None
             else torch.ones(mod.ternary.numel(), dtype=torch.bool, device=device))
            for mod in all_mods
        ])
        # Also check ternary bounds
        flat_ternary = torch.cat([mod.ternary.reshape(-1) for mod in all_mods])
        can_up = flat_alive & (flat_ternary < 1) & (flat_pressures > 0)
        can_down = flat_alive & (flat_ternary > -1) & (flat_pressures < 0)
        can_flip = can_up | can_down

        flat_abs = flat_pressures.abs()
        flat_abs[~can_flip] = -1  # exclude non-flippable

        if can_flip.sum() > K:
            topk_vals, topk_idx = flat_abs.topk(K)
            flip_mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            flip_mask[topk_idx] = True
        else:
            flip_mask = can_flip

        # Apply flips
        offset = 0
        for mod, p in zip(all_mods, all_pressures):
            n = mod.ternary.numel()
            local_flip = flip_mask[offset:offset+n].reshape(mod.ternary.shape)
            local_dir = (p > 0).to(torch.int8) * 2 - 1  # +1 or -1

            up = local_flip & (local_dir == 1) & (mod.ternary < 1)
            down = local_flip & (local_dir == -1) & (mod.ternary > -1)
            mod.ternary[up] += 1
            mod.ternary[down] -= 1
            p[up | down] = 0
            offset += n

        if step % 500 == 0:
            avg = running_loss / 500
            elapsed = time.time() - t_start
            print(f"  [exp1] step {step:>5d} | loss {avg:.4f} | {step/elapsed:.0f} step/s", flush=True)
            wandb.log({"train/loss": avg, "perf/step_s": step/elapsed}, step=step)
            running_loss = 0.0

        if step % 2500 == 0:
            pf_model.eval()
            val, ppl = evaluate(pf_model, device)
            print(f"  [exp1] EVAL {step}: val={val:.4f} ppl={ppl:.1f}", flush=True)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)

    pf_model.eval()
    val, ppl = evaluate(pf_model, device)
    print(f"  [exp1] FINAL: val={val:.4f} ppl={ppl:.1f} ({time.time()-t_start:.0f}s)")
    wandb.log({"val/loss": val, "val/ppl": ppl}, step=PHASE2_STEPS)
    wandb.finish()
    return val, ppl


# ===========================================================================
# Experiment 2: PF Annealed flip-rate (1000→50 cosine)
# ===========================================================================

def run_exp2(device):
    """Annealed Top-K: start aggressive (1000/step), end conservative (50/step)."""
    from pressure_flip import create_pressure_flip_model
    model, cfg = load_phase1(device)
    pruned_data, density = prune_model(model, device)
    pf_model = create_pressure_flip_model(model, pruned_data, cfg, device)
    del model; torch.cuda.empty_cache()

    BETA = 0.9
    K_START = 1000
    K_END = 50

    wandb.init(project=WANDB_PROJECT, name=f"overnight_pf_anneal_{K_START}to{K_END}",
               settings=wandb.Settings(init_timeout=180),
               config={"exp": 2, "method": "pf_annealed", "K_start": K_START, "K_end": K_END})

    pressures = {}
    for name, mod in pf_model.pf_modules:
        pressures[name] = torch.zeros(mod.out_features, mod.in_features,
                                      dtype=torch.bfloat16, device=device)

    val0, ppl0 = evaluate(pf_model, device)
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)
    print(f"  [exp2] start: val={val0:.4f} ppl={ppl0:.1f}")

    loader = get_loader()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break
        x, y = x.to(device), y.to(device)

        # Cosine annealed K
        progress = step / PHASE2_STEPS
        K = int(K_END + 0.5 * (K_START - K_END) * (1 + math.cos(math.pi * progress)))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pf_model(x, labels=y)
        out["loss"].backward()
        running_loss += out["loss"].item()

        # Accumulate + Top-K flip (same logic as exp1 but with variable K)
        all_pressures = []
        all_mods = []
        for name, mod in pf_model.pf_modules:
            if mod._w_leaf is None or mod._w_leaf.grad is None:
                continue
            grad = mod._w_leaf.grad
            if mod.weight_mask is not None:
                grad = grad * mod.weight_mask
            p = pressures[name]
            p.mul_(BETA).add_(grad.to(torch.bfloat16), alpha=-(1 - BETA))
            all_pressures.append(p)
            all_mods.append(mod)
            mod._w_leaf = None

        flat_pressures = torch.cat([p.reshape(-1) for p in all_pressures])
        flat_alive = torch.cat([
            (mod.weight_mask.reshape(-1).bool() if mod.weight_mask is not None
             else torch.ones(mod.ternary.numel(), dtype=torch.bool, device=device))
            for mod in all_mods
        ])
        flat_ternary = torch.cat([mod.ternary.reshape(-1) for mod in all_mods])
        can_up = flat_alive & (flat_ternary < 1) & (flat_pressures > 0)
        can_down = flat_alive & (flat_ternary > -1) & (flat_pressures < 0)
        can_flip = can_up | can_down

        flat_abs = flat_pressures.abs()
        flat_abs[~can_flip] = -1

        actual_K = min(K, can_flip.sum().item())
        if actual_K > 0:
            topk_vals, topk_idx = flat_abs.topk(actual_K)
            flip_mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            flip_mask[topk_idx] = True
        else:
            flip_mask = torch.zeros_like(flat_abs, dtype=torch.bool)

        offset = 0
        for mod, p in zip(all_mods, all_pressures):
            n = mod.ternary.numel()
            local_flip = flip_mask[offset:offset+n].reshape(mod.ternary.shape)
            local_dir = (p > 0).to(torch.int8) * 2 - 1
            up = local_flip & (local_dir == 1) & (mod.ternary < 1)
            down = local_flip & (local_dir == -1) & (mod.ternary > -1)
            mod.ternary[up] += 1
            mod.ternary[down] -= 1
            p[up | down] = 0
            offset += n

        if step % 500 == 0:
            avg = running_loss / 500
            elapsed = time.time() - t_start
            print(f"  [exp2] step {step:>5d} | loss {avg:.4f} | K={K} | {step/elapsed:.0f} step/s", flush=True)
            wandb.log({"train/loss": avg, "pf/K": K}, step=step)
            running_loss = 0.0

        if step % 2500 == 0:
            pf_model.eval()
            val, ppl = evaluate(pf_model, device)
            print(f"  [exp2] EVAL {step}: val={val:.4f} ppl={ppl:.1f}", flush=True)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)

    pf_model.eval()
    val, ppl = evaluate(pf_model, device)
    print(f"  [exp2] FINAL: val={val:.4f} ppl={ppl:.1f} ({time.time()-t_start:.0f}s)")
    wandb.log({"val/loss": val, "val/ppl": ppl}, step=PHASE2_STEPS)
    wandb.finish()
    return val, ppl


# ===========================================================================
# Experiment 3: PF bf16 momentum, low threshold, no cooldown
# ===========================================================================

def run_exp3(device):
    """bf16 momentum with FLIP_MULT=0.05 (10x lower than before), no cooldown."""
    from pressure_flip import create_pressure_flip_model
    model, cfg = load_phase1(device)
    pruned_data, density = prune_model(model, device)
    pf_model = create_pressure_flip_model(model, pruned_data, cfg, device)
    del model; torch.cuda.empty_cache()

    BETA = 0.9
    LR_SCALE = 1.0
    FLIP_MULT = 0.05

    wandb.init(project=WANDB_PROJECT, name=f"overnight_pf_lowthresh_fm{FLIP_MULT}",
               settings=wandb.Settings(init_timeout=180),
               config={"exp": 3, "method": "pf_lowthresh", "beta": BETA, "flip_mult": FLIP_MULT})

    pressures = {}
    for name, mod in pf_model.pf_modules:
        pressures[name] = torch.zeros(mod.out_features, mod.in_features,
                                      dtype=torch.bfloat16, device=device)

    val0, ppl0 = evaluate(pf_model, device)
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)
    print(f"  [exp3] start: val={val0:.4f} ppl={ppl0:.1f}")

    loader = get_loader()
    running_loss = 0.0
    total_flips = 0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break
        x, y = x.to(device), y.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pf_model(x, labels=y)
        out["loss"].backward()
        running_loss += out["loss"].item()

        step_flips = 0
        for name, mod in pf_model.pf_modules:
            if mod._w_leaf is None or mod._w_leaf.grad is None:
                continue
            grad = mod._w_leaf.grad
            if mod.weight_mask is not None:
                grad = grad * mod.weight_mask
            p = pressures[name]
            p.mul_(BETA).add_(grad.to(torch.bfloat16), alpha=-(1 - BETA) * LR_SCALE)

            # Scale-relative threshold
            gs = mod.group_size
            n_groups = mod.ternary.numel() // gs
            scales_exp = (mod.scales.unsqueeze(1).expand(n_groups, gs)
                          .reshape(mod.out_features, mod.in_features))
            threshold = FLIP_MULT * scales_exp

            flip_up = (p > threshold) & (mod.ternary < 1)
            flip_down = (p < -threshold) & (mod.ternary > -1)
            if mod.weight_mask is not None:
                alive = mod.weight_mask.bool()
                flip_up = flip_up & alive
                flip_down = flip_down & alive

            mod.ternary[flip_up] += 1
            mod.ternary[flip_down] -= 1
            flipped = flip_up | flip_down
            p[flipped] = 0
            step_flips += flipped.sum().item()
            mod._w_leaf = None

        total_flips += step_flips

        if step % 500 == 0:
            avg = running_loss / 500
            elapsed = time.time() - t_start
            avg_flips = total_flips / step
            print(f"  [exp3] step {step:>5d} | loss {avg:.4f} | flips {avg_flips:.0f}/step | "
                  f"{step/elapsed:.0f} step/s", flush=True)
            wandb.log({"train/loss": avg, "pf/flips_per_step": avg_flips}, step=step)
            running_loss = 0.0

        if step % 2500 == 0:
            pf_model.eval()
            val, ppl = evaluate(pf_model, device)
            print(f"  [exp3] EVAL {step}: val={val:.4f} ppl={ppl:.1f}", flush=True)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)

    pf_model.eval()
    val, ppl = evaluate(pf_model, device)
    print(f"  [exp3] FINAL: val={val:.4f} ppl={ppl:.1f} ({time.time()-t_start:.0f}s)")
    wandb.log({"val/loss": val, "val/ppl": ppl}, step=PHASE2_STEPS)
    wandb.finish()
    return val, ppl


# ===========================================================================
# Experiment 4: Hybrid STE 2k → PF 18k
# ===========================================================================

def run_exp4(device):
    """STE+Adam for 2k steps (find config), then PF int8 threshold=10 for 18k."""
    from model import TernaryTransformer
    from pressure_flip import create_pressure_flip_model, PressureFlipTrainer

    model, cfg = load_phase1(device)
    pruned_data, density = prune_model(model, device)

    wandb.init(project=WANDB_PROJECT, name="overnight_hybrid_ste2k_pf18k",
               settings=wandb.Settings(init_timeout=180),
               config={"exp": 4, "method": "hybrid_ste_pf", "ste_steps": 2000, "pf_steps": 18000})

    val0, ppl0 = evaluate(model, device)
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)
    print(f"  [exp4] start: val={val0:.4f} ppl={ppl0:.1f}")

    # Phase A: STE+Adam for 2k steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                  weight_decay=0.1, eps=1e-8)
    loader = get_loader()
    model.train()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > 2000:
            break
        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, WARMUP, 2000, LR, LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, labels=y)["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        if step % 500 == 0:
            avg = running_loss / 500
            print(f"  [exp4/ste] step {step} | loss {avg:.4f}", flush=True)
            wandb.log({"train/loss": avg, "phase": "ste"}, step=step)
            running_loss = 0.0

    val_ste, ppl_ste = evaluate(model, device)
    print(f"  [exp4] after STE 2k: val={val_ste:.4f} ppl={ppl_ste:.1f}")
    wandb.log({"val/loss": val_ste, "val/ppl": ppl_ste}, step=2000)

    # Re-prune from STE-refined model
    pruned_data2, _ = prune_model(model, device)

    # Phase B: PF for 18k steps
    pf_model = create_pressure_flip_model(model, pruned_data2, cfg, device)
    del model, optimizer; torch.cuda.empty_cache()

    trainer = PressureFlipTrainer(pf_model, threshold_start=10, threshold_end=10,
                                  threshold_schedule="constant", total_steps=18000)

    loader = get_loader()
    running_loss = 0.0

    for step, (x, y) in enumerate(loader, 1):
        if step > 18000:
            break
        stats = trainer.step(x, y, device)
        running_loss += stats["loss"]

        if step % 500 == 0:
            avg = running_loss / 500
            global_step = 2000 + step
            print(f"  [exp4/pf] step {global_step} | loss {avg:.4f} | flips {stats['n_flipped']}", flush=True)
            wandb.log({"train/loss": avg, "phase": "pf", "pf/flips": stats["n_flipped"]}, step=global_step)
            running_loss = 0.0

        if step % 2500 == 0:
            pf_model.eval()
            val, ppl = evaluate(pf_model, device)
            global_step = 2000 + step
            print(f"  [exp4] EVAL {global_step}: val={val:.4f} ppl={ppl:.1f}", flush=True)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=global_step)

    pf_model.eval()
    val, ppl = evaluate(pf_model, device)
    print(f"  [exp4] FINAL: val={val:.4f} ppl={ppl:.1f} ({time.time()-t_start:.0f}s)")
    wandb.log({"val/loss": val, "val/ppl": ppl}, step=20000)
    wandb.finish()
    return val, ppl


# ===========================================================================
# Experiment 5: STE + SGD momentum (no Adam) — 6 bytes/param
# ===========================================================================

def run_exp5(device):
    """Standard STE but SGD+momentum instead of Adam. Tests if Adam is needed."""
    from model import TernaryTransformer
    model, cfg = load_phase1(device)
    pruned_data, density = prune_model(model, device)

    wandb.init(project=WANDB_PROJECT, name="overnight_ste_sgd_mom09",
               settings=wandb.Settings(init_timeout=180),
               config={"exp": 5, "method": "ste_sgd", "lr": LR, "momentum": 0.9})

    val0, ppl0 = evaluate(model, device)
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)
    print(f"  [exp5] start: val={val0:.4f} ppl={ppl0:.1f}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.1)

    loader = get_loader()
    model.train()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break
        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, WARMUP, PHASE2_STEPS, LR, LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, labels=y)["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        if step % 500 == 0:
            avg = running_loss / 500
            elapsed = time.time() - t_start
            print(f"  [exp5] step {step:>5d} | loss {avg:.4f} | {step/elapsed:.0f} step/s", flush=True)
            wandb.log({"train/loss": avg}, step=step)
            running_loss = 0.0

        if step % 2500 == 0:
            val, ppl = evaluate(model, device)
            print(f"  [exp5] EVAL {step}: val={val:.4f} ppl={ppl:.1f}", flush=True)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)
            model.train()

    val, ppl = evaluate(model, device)
    print(f"  [exp5] FINAL: val={val:.4f} ppl={ppl:.1f} ({time.time()-t_start:.0f}s)")
    wandb.log({"val/loss": val, "val/ppl": ppl}, step=PHASE2_STEPS)
    wandb.finish()
    return val, ppl


# ===========================================================================
# Experiment 6: STE + Adam at bf16 optimizer precision — 8 bytes/param
# ===========================================================================

def run_exp6(device):
    """STE+Adam but force optimizer states to bf16. Tests precision requirements."""
    from model import TernaryTransformer
    model, cfg = load_phase1(device)
    pruned_data, density = prune_model(model, device)

    wandb.init(project=WANDB_PROJECT, name="overnight_ste_adam_bf16opt",
               settings=wandb.Settings(init_timeout=180),
               config={"exp": 6, "method": "ste_adam_bf16", "lr": LR})

    val0, ppl0 = evaluate(model, device)
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)
    print(f"  [exp6] start: val={val0:.4f} ppl={ppl0:.1f}")

    # Standard Adam but we'll manually cast states to bf16 after each step
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                  weight_decay=0.1, eps=1e-8)

    loader = get_loader()
    model.train()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break
        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, WARMUP, PHASE2_STEPS, LR, LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, labels=y)["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        # Quantize optimizer states to bf16 (simulate reduced precision)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                if "exp_avg" in state:
                    state["exp_avg"] = state["exp_avg"].to(torch.bfloat16).to(torch.float32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.bfloat16).to(torch.float32)

        if step % 500 == 0:
            avg = running_loss / 500
            elapsed = time.time() - t_start
            print(f"  [exp6] step {step:>5d} | loss {avg:.4f} | {step/elapsed:.0f} step/s", flush=True)
            wandb.log({"train/loss": avg}, step=step)
            running_loss = 0.0

        if step % 2500 == 0:
            val, ppl = evaluate(model, device)
            print(f"  [exp6] EVAL {step}: val={val:.4f} ppl={ppl:.1f}", flush=True)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)
            model.train()

    val, ppl = evaluate(model, device)
    print(f"  [exp6] FINAL: val={val:.4f} ppl={ppl:.1f} ({time.time()-t_start:.0f}s)")
    wandb.log({"val/loss": val, "val/ppl": ppl}, step=PHASE2_STEPS)
    wandb.finish()
    return val, ppl


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=0, help="Run specific exp (1-9), 0=all")
    args = parser.parse_args()
    device = torch.device("cuda")

    experiments = {
        1: ("PF Top-K 50", lambda d: run_exp1(d, K=50)),
        2: ("PF Top-K 200", lambda d: run_exp1(d, K=200)),
        3: ("PF Top-K 500", lambda d: run_exp1(d, K=500)),
        4: ("PF Top-K 1000", lambda d: run_exp1(d, K=1000)),
        5: ("PF Annealed 1000→50", run_exp2),
        6: ("PF bf16 low-thresh", run_exp3),
        7: ("Hybrid STE→PF", run_exp4),
        8: ("STE+SGD", run_exp5),
        9: ("STE+Adam bf16", run_exp6),
    }

    to_run = [args.exp] if args.exp > 0 else list(experiments.keys())
    results = {}

    print(f"\n{'='*60}")
    print(f"  OVERNIGHT PRESSURE FLIP EXPERIMENTS")
    print(f"  Running: {[experiments[i][0] for i in to_run]}")
    print(f"{'='*60}\n")

    for exp_id in to_run:
        name, fn = experiments[exp_id]
        print(f"\n{'─'*50}")
        print(f"  Experiment {exp_id}: {name}")
        print(f"{'─'*50}")
        try:
            val, ppl = fn(device)
            results[exp_id] = (name, val, ppl)
        except Exception as e:
            print(f"  ERROR in exp {exp_id}: {e}")
            import traceback; traceback.print_exc()
            results[exp_id] = (name, None, None)
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    print(f"  OVERNIGHT RESULTS")
    print(f"  {'#':<3} {'Method':<25} {'Val Loss':<10} {'Val PPL':<10}")
    print(f"  {'-'*55}")
    # Reference baselines
    print(f"  {'—':<3} {'bf16 (25k, ref)':<25} {'4.94':<10} {'140':<10}")
    print(f"  {'—':<3} {'STE+Adam (20k, ref)':<25} {'5.33':<10} {'206':<10}")
    print(f"  {'-'*55}")
    for exp_id in sorted(results.keys()):
        name, val, ppl = results[exp_id]
        val_s = f"{val:.4f}" if val else "FAILED"
        ppl_s = f"{ppl:.1f}" if ppl else "—"
        print(f"  {exp_id:<3} {name:<25} {val_s:<10} {ppl_s:<10}")
    print(f"{'='*60}")
