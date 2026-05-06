"""A/B/C/D Test: Ternary training methods after pruning.

All arms:
  1. Train Phase 1 (STE+Adam) OR load existing checkpoint
  2. Per-layer cold prune to target density
  3. Train for same number of steps on same data stream
  4. Log to wandb for comparison

Arms:
  A (baseline): STE+Adam continuation — backprop through ternary
  B (eggroll):  fast_es_step — no backprop, pure Evolution Strategies
  C (nes_cat):  NES-Categorical — score-function gradient on ternary distributions
  D (lrnet):    LR-nets — CLT reparameterization, backprop on distribution params

Usage:
  # Phase 1 + baseline
  python run_ab_eggroll.py --arm baseline --d_model 128 --n_layers 3 --phase1_steps 5000

  # NES-Categorical A4 (per-matrix cycling + momentum) with existing checkpoint
  python run_ab_eggroll.py --arm nes_cat --checkpoint checkpoints/ab_phase1_d128L3.pt \\
      --density 0.2 --nes_pop 256 --nes_cycling per_matrix --nes_momentum 0.9

  # NES-Categorical A1 (full-space, no momentum)
  python run_ab_eggroll.py --arm nes_cat --checkpoint checkpoints/ab_phase1_d128L3.pt \\
      --density 0.2 --nes_pop 2048 --nes_cycling full --nes_momentum 0.0

  # LR-nets B1
  python run_ab_eggroll.py --arm lrnet --checkpoint checkpoints/ab_phase1_d128L3.pt --density 0.2
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import wandb

from model import TernaryTransformer, TernaryConfig
from eggroll_model import EggrollTransformer, EggrollConfig, EggrollLinear, EggrollEmbedding
from bitlinear import BitLinear, BitEmbedding, ternary_quantize

torch.set_float32_matmul_precision('high')
os.environ["PYTHONUNBUFFERED"] = "1"

WANDB_PROJECT = "bonsai-ternary-test"


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

def get_data_loader(batch_size: int, seq_len: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    buffer = []
    for sample in ds:
        tokens = tokenizer(sample["text"], truncation=False, add_special_tokens=False)["input_ids"]
        buffer.extend(tokens)
        while len(buffer) >= batch_size * (seq_len + 1):
            batch_tokens = buffer[:batch_size * (seq_len + 1)]
            buffer = buffer[batch_size * (seq_len + 1):]
            t = torch.tensor(batch_tokens, dtype=torch.long).reshape(batch_size, seq_len + 1)
            yield t[:, :seq_len], t[:, 1:seq_len + 1]


def cosine_lr(step, warmup, total, lr_max, lr_min=0.0):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def eval_model(model, device, n_batches=10, seq_len=256, batch_size=4):
    model.eval()
    loader = get_data_loader(batch_size, seq_len)
    losses = []
    with torch.no_grad():
        for i, (vx, vy) in enumerate(loader):
            if i >= n_batches:
                break
            vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(vx, labels=vy)
            losses.append(out["loss"].item())
    if not losses:
        return 20.0, math.exp(20)
    val_loss = sum(losses) / len(losses)
    return val_loss, math.exp(min(val_loss, 20))


def perlayer_prune(model, density, device, skip_embed_lmhead=False):
    """Per-layer cold pruning by shadow weight magnitude.

    Args:
        skip_embed_lmhead: If True, keep embed and lm_head at 100% density.
            Use this for EGGROLL which doesn't optimize those layers.
    """
    print(f"\n  Pruning to {density*100:.0f}% density (per-layer)"
          + (" [skip embed/lm_head]" if skip_embed_lmhead else ""))
    pruned_data = {}
    total_alive = 0
    total_params = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, (BitLinear, BitEmbedding)):
            continue

        # Optionally skip embedding and lm_head
        if skip_embed_lmhead and (name in ("embed", "lm_head") or isinstance(mod, BitEmbedding)):
            shadow = mod.weight.detach().float()
            gs = mod.group_size
            total_el = shadow.numel()
            while total_el % gs != 0 and gs > 1:
                gs -= 1
            w_q, scales = ternary_quantize(shadow, gs)
            mask = torch.ones_like(shadow, device=device)
            pruned_data[name] = {
                "ternary": w_q.to(torch.int8).to(device),
                "scales": scales.to(torch.bfloat16).to(device),
                "mask": mask, "group_size": gs, "shape": tuple(shadow.shape),
            }
            total_alive += shadow.numel()
            total_params += shadow.numel()
            print(f"    {name}: {shadow.numel()}/{shadow.numel()} alive (100.0%) [KEPT]")
            continue

        shadow = mod.weight.detach().float()
        flat_mag = shadow.reshape(-1).abs()
        n = flat_mag.numel()
        n_keep = max(1, int(n * density))

        if n_keep < n:
            threshold = flat_mag.kthvalue(n - n_keep).values.item()
        else:
            threshold = 0.0
        mask = (shadow.abs() >= threshold).float().to(device)

        gs = mod.group_size
        total_el = shadow.numel()
        while total_el % gs != 0 and gs > 1:
            gs -= 1
        w_q, scales = ternary_quantize(shadow, gs)
        w_q = (w_q.float() * mask).to(torch.int8).to(device)
        scales = scales.to(torch.bfloat16).to(device)

        alive = mask.bool().sum().item()
        total_alive += alive
        total_params += n

        mod.weight.data *= mask

        pruned_data[name] = {
            "ternary": w_q, "scales": scales, "mask": mask,
            "group_size": gs, "shape": tuple(shadow.shape),
        }
        print(f"    {name}: {alive}/{n} alive ({100*alive/n:.1f}%)")

    actual_density = total_alive / max(1, total_params)
    print(f"    Overall: {actual_density*100:.1f}% ({total_alive/1e6:.2f}M alive)")
    return pruned_data, actual_density


# ---------------------------------------------------------------------------
# Phase 1: STE+Adam training
# ---------------------------------------------------------------------------

def train_phase1(args, device):
    """Train Phase 1 STE model from scratch."""
    d_ff = int(args.d_model * 2.69)
    # Round d_ff to multiple of 8 for efficiency
    d_ff = ((d_ff + 7) // 8) * 8

    cfg = TernaryConfig(
        d_model=args.d_model, n_heads=max(1, args.d_model // 64),
        d_ff=d_ff, n_layers=args.n_layers,
        vocab_size=49152, max_seq_len=1024,
        group_size=min(128, args.d_model),
    )
    model = TernaryTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Phase 1 model: {n_params/1e6:.2f}M params, d={cfg.d_model}, L={cfg.n_layers}, d_ff={cfg.d_ff}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-8,
    )

    loader = get_data_loader(args.phase1_batch_size, args.seq_len)
    model.train()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > args.phase1_steps:
            break

        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, 200, args.phase1_steps, args.lr, args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
            loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        if step % 100 == 0:
            avg_loss = running_loss / 100
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            print(f"    [P1] step {step:>5d}/{args.phase1_steps} | loss {avg_loss:.4f} | "
                  f"ppl {ppl:.1f} | {step/elapsed:.1f} step/s", flush=True)
            wandb.log({"p1/loss": avg_loss, "p1/ppl": ppl, "p1/lr": lr}, step=step)
            running_loss = 0.0

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_name = f"ab_phase1_d{args.d_model}L{args.n_layers}"
    ckpt_path = f"checkpoints/{ckpt_name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"    Phase 1 saved: {ckpt_path}")

    # Eval
    val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    Phase 1 val: loss={val_loss:.4f}, ppl={val_ppl:.1f}")

    elapsed = time.time() - t_start
    print(f"    Phase 1 done: {args.phase1_steps} steps in {elapsed:.0f}s")

    return model, cfg, ckpt_path


# ---------------------------------------------------------------------------
# Arm A: Baseline — continued STE+Adam
# ---------------------------------------------------------------------------

def run_baseline(args, p1_model, p1_cfg, pruned_data, actual_density, device):
    print(f"\n  ARM A: BASELINE (STE+Adam, {args.steps} steps)")
    # p1_model already has pruned shadow weights from perlayer_prune

    val_loss0, val_ppl0 = eval_model(p1_model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    Post-prune val: loss={val_loss0:.4f}, ppl={val_ppl0:.1f}")

    wandb.init(
        project=WANDB_PROJECT,
        name=f"ab_baseline_d{args.d_model}L{args.n_layers}_d{int(args.density*100)}",
        settings=wandb.Settings(init_timeout=180),
        config={
            "arm": "baseline", "density": actual_density,
            "steps": args.steps, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "d_model": args.d_model,
            "n_layers": args.n_layers, "lr": args.lr,
        },
    )
    wandb.log({"val/loss": val_loss0, "val/ppl": val_ppl0}, step=0)

    optimizer = torch.optim.AdamW(
        p1_model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-8,
    )

    p1_model.train()
    loader = get_data_loader(args.batch_size, args.seq_len)
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > args.steps:
            break

        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, 200, args.steps, args.lr, args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = p1_model(x, labels=y)
            loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p1_model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        if step % 50 == 0:
            avg_loss = running_loss / 50
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            print(f"    [base] step {step:>5d}/{args.steps} | loss {avg_loss:.4f} | "
                  f"ppl {ppl:.1f} | {step/elapsed:.1f} step/s", flush=True)
            wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "lr": lr}, step=step)
            running_loss = 0.0

        if step % 500 == 0:
            val_loss, val_ppl = eval_model(p1_model, device, seq_len=args.seq_len, batch_size=args.batch_size)
            print(f"    [base] EVAL step {step}: val_loss={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)
            p1_model.train()

    # Final
    val_loss, val_ppl = eval_model(p1_model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    [base] FINAL: val_loss={val_loss:.4f} ppl={val_ppl:.1f}")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=args.steps)

    elapsed = time.time() - t_start
    print(f"    Baseline done: {args.steps} steps in {elapsed:.0f}s ({args.steps/elapsed:.1f} step/s)")
    wandb.finish()
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Arm B: EGGROLL
# ---------------------------------------------------------------------------

def run_eggroll(args, p1_model, p1_cfg, pruned_data, actual_density, device):
    print(f"\n  ARM B: EGGROLL (pop={args.pop_size}, {args.steps} steps)")

    d_ff = p1_cfg.d_ff if hasattr(p1_cfg, 'd_ff') else int(p1_cfg.d_model * 2.69)
    egg_cfg = EggrollConfig(
        d_model=p1_cfg.d_model, n_heads=p1_cfg.n_heads, d_ff=d_ff,
        n_layers=p1_cfg.n_layers, vocab_size=p1_cfg.vocab_size,
        max_seq_len=p1_cfg.max_seq_len,
        group_size=p1_cfg.group_size if hasattr(p1_cfg, 'group_size') else 128,
        pop_size=args.pop_size, sigma=args.sigma, es_lr=args.es_lr,
    )
    model = EggrollTransformer(egg_cfg).to(device)
    _transfer_pruned_weights(model, pruned_data, p1_model, device)

    # Count alive linear params (excluding embed/lm_head which ES doesn't touch)
    n_alive_linear = 0
    n_total_linear = 0
    for name, mod, _ in model._es_modules:
        if mod is model.embed or mod is model.lm_head:
            continue
        alive = mod.weight_mask.sum().item() if mod.weight_mask is not None else mod.ternary.numel()
        n_alive_linear += alive
        n_total_linear += mod.ternary.numel()

    pop_ratio = n_alive_linear / args.pop_size
    print(f"    ES target: {n_alive_linear/1e3:.1f}k alive linear params")
    print(f"    Pop:param ratio = 1:{pop_ratio:.0f}")
    if pop_ratio > 1000:
        print(f"    ⚠ WARNING: ratio > 1:1000 — ES gradient will be very noisy!")

    model.eval()
    val_loss0, val_ppl0 = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    Post-transfer val: loss={val_loss0:.4f}, ppl={val_ppl0:.1f}")

    from kernels.eggroll_fast import FastPopulationEvaluator, fast_es_step
    evaluator = FastPopulationEvaluator(model, use_chunked_ce=True, compile=True)

    # Warmup compile
    print("    Compiling...")
    x_w = torch.randint(0, egg_cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    y_w = torch.randint(0, egg_cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    fast_es_step(model, x_w, y_w, step=0, evaluator=evaluator)
    torch.cuda.synchronize()
    del x_w, y_w
    torch.cuda.empty_cache()
    print("    Compiled!")

    wandb.init(
        project=WANDB_PROJECT,
        name=f"ab_eggroll_d{args.d_model}L{args.n_layers}_d{int(args.density*100)}_p{args.pop_size}",
        settings=wandb.Settings(init_timeout=180),
        config={
            "arm": "eggroll", "density": actual_density,
            "steps": args.steps, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "d_model": args.d_model,
            "n_layers": args.n_layers,
            "pop_size": args.pop_size, "sigma": args.sigma,
            "es_lr": args.es_lr, "n_alive_linear": n_alive_linear,
            "pop_param_ratio": f"1:{pop_ratio:.0f}",
        },
    )
    wandb.log({"val/loss": val_loss0, "val/ppl": val_ppl0}, step=0)

    loader = get_data_loader(args.batch_size, args.seq_len)
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > args.steps:
            break

        x, y = x.to(device), y.to(device)
        stats = fast_es_step(model, x, y, step=step, evaluator=evaluator)

        with torch.no_grad():
            out = model(x, labels=y)
        running_loss += out["loss"].item()

        if step % 50 == 0:
            avg_loss = running_loss / 50
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = step / elapsed
            eta = (args.steps - step) / steps_per_s if steps_per_s > 0 else 0

            print(f"    [egg] step {step:>5d}/{args.steps} | loss {avg_loss:.4f} | "
                  f"ppl {ppl:.1f} | {steps_per_s:.2f} step/s | "
                  f"flips {stats['n_flips']} | ETA {eta/60:.0f}min", flush=True)

            tstats = model.ternary_stats()
            wandb.log({
                "train/loss": avg_loss, "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "flip/n_flips": stats["n_flips"],
                "ternary/zero_frac": tstats["zero_frac"],
            }, step=step)
            running_loss = 0.0

        if step % 500 == 0:
            val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
            print(f"    [egg] EVAL step {step}: val_loss={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

    # Final
    val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    [egg] FINAL: val_loss={val_loss:.4f} ppl={val_ppl:.1f}")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=args.steps)

    elapsed = time.time() - t_start
    print(f"    EGGROLL done: {args.steps} steps in {elapsed:.0f}s ({args.steps/elapsed:.2f} step/s)")
    wandb.finish()
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Arm C: NES-Categorical
# ---------------------------------------------------------------------------

def run_nes_cat(args, p1_model, p1_cfg, pruned_data, actual_density, device):
    """NES-Categorical arm: score-function gradient on ternary distributions."""
    print(f"\n  ARM C: NES-CAT (pop={args.nes_pop}, cycling={args.nes_cycling}, "
          f"mom={args.nes_momentum}, {args.steps} steps)")

    d_ff = p1_cfg.d_ff if hasattr(p1_cfg, 'd_ff') else int(p1_cfg.d_model * 2.69)
    egg_cfg = EggrollConfig(
        d_model=p1_cfg.d_model, n_heads=p1_cfg.n_heads, d_ff=d_ff,
        n_layers=p1_cfg.n_layers, vocab_size=p1_cfg.vocab_size,
        max_seq_len=p1_cfg.max_seq_len,
        group_size=p1_cfg.group_size if hasattr(p1_cfg, 'group_size') else 128,
    )
    model = EggrollTransformer(egg_cfg).to(device)
    _transfer_pruned_weights(model, pruned_data, p1_model, device)

    model.eval()
    val_loss0, val_ppl0 = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    Post-transfer val: loss={val_loss0:.4f}, ppl={val_ppl0:.1f}")

    from nes_categorical import NESCategoricalTrainer
    trainer = NESCategoricalTrainer(
        model,
        pop_size=args.nes_pop,
        momentum=args.nes_momentum,
        lr=args.nes_lr,
        tau_start=args.nes_tau_start,
        tau_end=args.nes_tau_end,
        total_steps=args.steps,
        cycling=args.nes_cycling,
        natural_gradient=args.nes_natural_gradient,
        fitness_shaping=not args.nes_no_fitness_shaping,
    )

    run_name = (f"nes_{args.nes_cycling}_p{args.nes_pop}_m{args.nes_momentum}"
                f"_d{args.d_model}L{args.n_layers}_d{int(args.density*100)}")
    if args.nes_natural_gradient:
        run_name += "_natgrad"

    wandb.init(
        project=WANDB_PROJECT, name=run_name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "arm": "nes_cat", "density": actual_density,
            "steps": args.steps, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "d_model": args.d_model,
            "n_layers": args.n_layers, "pop_size": args.nes_pop,
            "momentum": args.nes_momentum, "lr": args.nes_lr,
            "cycling": args.nes_cycling,
            "natural_gradient": args.nes_natural_gradient,
            "tau_start": args.nes_tau_start, "tau_end": args.nes_tau_end,
        },
    )
    wandb.log({"val/loss": val_loss0, "val/ppl": val_ppl0}, step=0)

    loader = get_data_loader(args.batch_size, args.seq_len)
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > args.steps:
            break

        x, y = x.to(device), y.to(device)
        stats = trainer.step(x, y)
        running_loss += stats["loss"]

        if step % 50 == 0:
            avg_loss = running_loss / 50
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = step / elapsed
            eta = (args.steps - step) / steps_per_s if steps_per_s > 0 else 0

            print(f"    [nes] step {step:>5d}/{args.steps} | loss {avg_loss:.4f} | "
                  f"ppl {ppl:.1f} | {steps_per_s:.1f} step/s | "
                  f"tau {stats['tau']:.3f} | ETA {eta/60:.0f}min", flush=True)

            dist_stats = trainer.distribution_stats()
            wandb.log({
                "train/loss": avg_loss, "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "nes/tau": stats["tau"],
                "nes/group": stats["group"],
                "nes/mean_entropy": dist_stats["mean_entropy"],
            }, step=step)
            running_loss = 0.0

        if step % 500 == 0:
            val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
            print(f"    [nes] EVAL step {step}: val_loss={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

    val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    [nes] FINAL: val_loss={val_loss:.4f} ppl={val_ppl:.1f}")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=args.steps)

    elapsed = time.time() - t_start
    print(f"    NES-Cat done: {args.steps} steps in {elapsed:.0f}s ({args.steps/elapsed:.1f} step/s)")
    wandb.finish()
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Arm D: LR-nets
# ---------------------------------------------------------------------------

def run_lrnet(args, p1_model, p1_cfg, pruned_data, actual_density, device):
    """LR-nets arm: CLT reparameterization, backprop on distribution params."""
    print(f"\n  ARM D: LRNET (lr={args.lrnet_lr}, entropy_reg={args.lrnet_entropy_reg}, "
          f"{args.steps} steps)")

    from lrnet import create_lrnet_model, LRNetTrainer

    model = create_lrnet_model(p1_model, pruned_data, p1_cfg, device)

    model.eval()
    val_loss0, val_ppl0 = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    Post-replacement val: loss={val_loss0:.4f}, ppl={val_ppl0:.1f}")

    trainer = LRNetTrainer(
        model,
        lr=args.lrnet_lr,
        entropy_reg=args.lrnet_entropy_reg,
        total_steps=args.steps,
    )

    run_name = (f"lrnet_d{args.d_model}L{args.n_layers}_d{int(args.density*100)}"
                f"_lr{args.lrnet_lr}")
    if args.lrnet_entropy_reg > 0:
        run_name += f"_ent{args.lrnet_entropy_reg}"

    wandb.init(
        project=WANDB_PROJECT, name=run_name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "arm": "lrnet", "density": actual_density,
            "steps": args.steps, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "d_model": args.d_model,
            "n_layers": args.n_layers, "lr": args.lrnet_lr,
            "entropy_reg": args.lrnet_entropy_reg,
        },
    )
    wandb.log({"val/loss": val_loss0, "val/ppl": val_ppl0}, step=0)

    loader = get_data_loader(args.batch_size, args.seq_len)
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > args.steps:
            break

        stats = trainer.step(x, y, device)
        running_loss += stats["loss"]

        if step % 50 == 0:
            avg_loss = running_loss / 50
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = step / elapsed

            print(f"    [lrn] step {step:>5d}/{args.steps} | loss {avg_loss:.4f} | "
                  f"ppl {ppl:.1f} | {steps_per_s:.1f} step/s | "
                  f"lr {stats['lr']:.2e}", flush=True)

            dist_stats = trainer.distribution_stats()
            wandb.log({
                "train/loss": avg_loss, "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "lr": stats["lr"],
                "lrnet/mean_entropy": dist_stats["mean_entropy"],
            }, step=step)
            running_loss = 0.0

        if step % 500 == 0:
            val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
            print(f"    [lrn] EVAL step {step}: val_loss={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)
            model.train()

    model.eval()
    val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    [lrn] FINAL: val_loss={val_loss:.4f} ppl={val_ppl:.1f}")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=args.steps)

    elapsed = time.time() - t_start
    print(f"    LRNet done: {args.steps} steps in {elapsed:.0f}s ({args.steps/elapsed:.1f} step/s)")
    wandb.finish()
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Arm E: Pressure Flip
# ---------------------------------------------------------------------------

def run_pressure_flip(args, p1_model, p1_cfg, pruned_data, actual_density, device):
    """Pressure Flip arm: sign accumulation + threshold flipping, 2 bytes/param."""
    print(f"\n  ARM E: PRESSURE FLIP (threshold={args.pf_threshold_start}->{args.pf_threshold_end}, "
          f"{args.pf_threshold_schedule}, {args.steps} steps)")

    from pressure_flip import create_pressure_flip_model, PressureFlipTrainer

    model = create_pressure_flip_model(p1_model, pruned_data, p1_cfg, device)

    model.eval()
    val_loss0, val_ppl0 = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    Post-build val: loss={val_loss0:.4f}, ppl={val_ppl0:.1f}")

    trainer = PressureFlipTrainer(
        model,
        threshold_start=args.pf_threshold_start,
        threshold_end=args.pf_threshold_end,
        threshold_schedule=args.pf_threshold_schedule,
        scale_lr=args.pf_scale_lr,
        total_steps=args.steps,
    )

    run_name = (f"pf_t{args.pf_threshold_start}-{args.pf_threshold_end}"
                f"_d{args.d_model}L{args.n_layers}_d{int(args.density*100)}")

    wandb.init(
        project=WANDB_PROJECT, name=run_name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "arm": "pressure_flip", "density": actual_density,
            "steps": args.steps, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "d_model": args.d_model,
            "n_layers": args.n_layers,
            "threshold_start": args.pf_threshold_start,
            "threshold_end": args.pf_threshold_end,
            "threshold_schedule": args.pf_threshold_schedule,
            "scale_lr": args.pf_scale_lr,
        },
    )
    wandb.log({"val/loss": val_loss0, "val/ppl": val_ppl0}, step=0)

    loader = get_data_loader(args.batch_size, args.seq_len)
    running_loss = 0.0
    running_flips = 0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > args.steps:
            break

        stats = trainer.step(x, y, device)
        running_loss += stats["loss"]
        running_flips += stats["n_flipped"]

        if step % 50 == 0:
            avg_loss = running_loss / 50
            avg_flips = running_flips / 50
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = step / elapsed

            print(f"    [pf] step {step:>5d}/{args.steps} | loss {avg_loss:.4f} | "
                  f"ppl {ppl:.1f} | {steps_per_s:.1f} step/s | "
                  f"flips {avg_flips:.0f}/step | thresh={stats['threshold']} | "
                  f"pressure={stats['mean_pressure']:.1f}", flush=True)

            wandb.log({
                "train/loss": avg_loss, "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "pf/flips_per_step": avg_flips,
                "pf/threshold": stats["threshold"],
                "pf/mean_pressure": stats["mean_pressure"],
            }, step=step)
            running_loss = 0.0
            running_flips = 0

        if step % 500 == 0:
            model.eval()
            val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
            print(f"    [pf] EVAL step {step}: val_loss={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

    model.eval()
    val_loss, val_ppl = eval_model(model, device, seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"    [pf] FINAL: val_loss={val_loss:.4f} ppl={val_ppl:.1f}")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=args.steps)

    elapsed = time.time() - t_start
    print(f"    PressureFlip done: {args.steps} steps in {elapsed:.0f}s ({args.steps/elapsed:.1f} step/s)")
    wandb.finish()
    return val_loss, val_ppl


def _transfer_pruned_weights(egg_model, pruned_data, p1_model, device):
    """Transfer ternary weights, scales, masks, norms, and alphas."""
    egg_modules = {n: m for n, m in egg_model.named_modules()
                   if isinstance(m, (EggrollLinear, EggrollEmbedding))}

    transferred = 0
    for name, data in pruned_data.items():
        if name not in egg_modules:
            continue
        emod = egg_modules[name]
        if emod.ternary.shape != data["ternary"].shape:
            continue
        emod.ternary.copy_(data["ternary"])
        emod.scales.copy_(data["scales"].to(emod.scales.dtype))
        emod.weight_mask = data["mask"].to(device)
        emod.group_size = data["group_size"]
        transferred += 1
    print(f"    Transferred {transferred}/{len(pruned_data)} layers")

    # Norms
    p1_norms = {n: m.scale.detach() for n, m in p1_model.named_modules()
                if hasattr(m, "scale") and isinstance(getattr(m, "scale"), torch.nn.Parameter)}
    egg_norms = {n: m.scale for n, m in egg_model.named_modules()
                 if hasattr(m, "scale") and isinstance(getattr(m, "scale"), torch.nn.Parameter)}
    for name, scale in p1_norms.items():
        if name in egg_norms and egg_norms[name].shape == scale.shape:
            egg_norms[name].data.copy_(scale.to(egg_norms[name].dtype))

    # Attention alpha
    for (n1, m1), (n2, m2) in zip(
        [(n, m) for n, m in p1_model.named_modules() if hasattr(m, "alpha")],
        [(n, m) for n, m in egg_model.named_modules() if hasattr(m, "alpha")],
    ):
        if hasattr(m1, "alpha") and hasattr(m2, "alpha"):
            if m1.alpha is not None and m2.alpha is not None:
                if m1.alpha.shape == m2.alpha.shape:
                    m2.alpha.copy_(m1.alpha.detach().to(m2.alpha.dtype))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ternary Training Method Comparison")
    parser.add_argument("--arm", choices=[
        "baseline", "eggroll", "nes_cat", "lrnet", "pressure_flip", "both",
    ], required=True)
    # Model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    # Phase 1
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Phase 1 checkpoint (skips Phase 1 training)")
    parser.add_argument("--phase1_steps", type=int, default=5000)
    parser.add_argument("--phase1_batch_size", type=int, default=12)
    # Shared
    parser.add_argument("--density", type=float, default=0.80)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4)
    # EGGROLL
    parser.add_argument("--pop_size", type=int, default=2048)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--es_lr", type=float, default=0.01)
    # NES-Categorical
    parser.add_argument("--nes_pop", type=int, default=256)
    parser.add_argument("--nes_momentum", type=float, default=0.9)
    parser.add_argument("--nes_lr", type=float, default=0.1)
    parser.add_argument("--nes_cycling", choices=["per_matrix", "per_layer", "full"],
                        default="per_matrix")
    parser.add_argument("--nes_natural_gradient", action="store_true")
    parser.add_argument("--nes_no_fitness_shaping", action="store_true")
    parser.add_argument("--nes_tau_start", type=float, default=2.0)
    parser.add_argument("--nes_tau_end", type=float, default=0.05)
    # LR-nets
    parser.add_argument("--lrnet_lr", type=float, default=3e-4)
    parser.add_argument("--lrnet_entropy_reg", type=float, default=0.0)
    # Pressure Flip
    parser.add_argument("--pf_threshold_start", type=int, default=10)
    parser.add_argument("--pf_threshold_end", type=int, default=40)
    parser.add_argument("--pf_threshold_schedule", choices=["cosine", "linear", "constant"],
                        default="cosine")
    parser.add_argument("--pf_scale_lr", type=float, default=0.0)

    args = parser.parse_args()
    device = torch.device("cuda")

    print(f"{'='*60}")
    print(f"  TERNARY TRAINING COMPARISON")
    print(f"  arm={args.arm}, d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"  density={args.density}, steps={args.steps}")
    if args.arm == "nes_cat":
        print(f"  NES: pop={args.nes_pop}, cycling={args.nes_cycling}, "
              f"mom={args.nes_momentum}, lr={args.nes_lr}")
    elif args.arm == "lrnet":
        print(f"  LRNet: lr={args.lrnet_lr}, entropy_reg={args.lrnet_entropy_reg}")
    elif args.arm == "pressure_flip":
        print(f"  PressureFlip: threshold={args.pf_threshold_start}->{args.pf_threshold_end} "
              f"({args.pf_threshold_schedule})")
    elif args.arm in ("eggroll", "both"):
        print(f"  EGGROLL: pop={args.pop_size}, sigma={args.sigma}, es_lr={args.es_lr}")
    print(f"{'='*60}")

    # Phase 1
    if args.checkpoint:
        print(f"\n  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        p1_cfg = ckpt["config"]
        p1_model = TernaryTransformer(p1_cfg).to(device)
        p1_model.load_state_dict(ckpt["model_state_dict"])
        n_params = sum(p.numel() for p in p1_model.parameters())
        print(f"  Loaded: {n_params/1e6:.2f}M params")
        ckpt_path = args.checkpoint
        args.d_model = p1_cfg.d_model
        args.n_layers = p1_cfg.n_layers
    else:
        wandb.init(project=WANDB_PROJECT,
                   name=f"ab_phase1_d{args.d_model}L{args.n_layers}",
                   settings=wandb.Settings(init_timeout=180),
                   config={"phase": 1, "d_model": args.d_model, "n_layers": args.n_layers})
        p1_model, p1_cfg, ckpt_path = train_phase1(args, device)
        wandb.finish()

    p1_state = {k: v.clone() for k, v in p1_model.state_dict().items()}

    results = {}

    if args.arm in ("baseline", "both"):
        pruned_data, actual_density = perlayer_prune(p1_model, args.density, device)
        bl_loss, bl_ppl = run_baseline(args, p1_model, p1_cfg, pruned_data, actual_density, device)
        results["baseline"] = bl_ppl

    if args.arm in ("eggroll", "both"):
        p1_model.load_state_dict(p1_state)
        pruned_data, actual_density = perlayer_prune(
            p1_model, args.density, device, skip_embed_lmhead=True)
        eg_loss, eg_ppl = run_eggroll(args, p1_model, p1_cfg, pruned_data, actual_density, device)
        results["eggroll"] = eg_ppl

    if args.arm == "nes_cat":
        p1_model.load_state_dict(p1_state)
        pruned_data, actual_density = perlayer_prune(
            p1_model, args.density, device, skip_embed_lmhead=True)
        nc_loss, nc_ppl = run_nes_cat(args, p1_model, p1_cfg, pruned_data, actual_density, device)
        results["nes_cat"] = nc_ppl

    if args.arm == "lrnet":
        p1_model.load_state_dict(p1_state)
        pruned_data, actual_density = perlayer_prune(
            p1_model, args.density, device, skip_embed_lmhead=True)
        lr_loss, lr_ppl = run_lrnet(args, p1_model, p1_cfg, pruned_data, actual_density, device)
        results["lrnet"] = lr_ppl

    if args.arm == "pressure_flip":
        p1_model.load_state_dict(p1_state)
        pruned_data, actual_density = perlayer_prune(
            p1_model, args.density, device, skip_embed_lmhead=True)
        pf_loss, pf_ppl = run_pressure_flip(args, p1_model, p1_cfg, pruned_data, actual_density, device)
        results["pressure_flip"] = pf_ppl

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  RESULTS")
        for name, ppl in results.items():
            print(f"  {name:>12s}: val_ppl = {ppl:.1f}")
        print(f"{'='*60}")
