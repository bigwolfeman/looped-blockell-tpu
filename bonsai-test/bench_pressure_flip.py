"""Head-to-head benchmark: bf16 vs STE+Adam vs Pressure Flip.

All three train for 25k total steps under identical conditions:
  - Architecture: d=128, 3 layers, d_ff=344, vocab=49152
  - Data: OpenWebText streaming, B=8, seq=256
  - LR: 6e-4 with cosine decay to 6e-5, 500 step warmup

Pipeline for ternary methods:
  Phase 1 (shared): STE+Adam at full density, 5k steps → checkpoint
  Phase 2 (diverge): Prune to 20% density, then train 20k steps with:
    - STE+Adam (standard, 14 bytes/param)
    - Pressure Flip (2 bytes/param)

bf16 baseline trains from scratch for 25k steps (no ternary, no pruning).

Usage:
  python bench_pressure_flip.py              # run all three
  python bench_pressure_flip.py --arm bf16   # just bf16
  python bench_pressure_flip.py --arm pf     # just pressure flip (uses shared Phase 1)
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
BATCH_SIZE = 8
SEQ_LEN = 256
D_MODEL = 128
N_LAYERS = 3
VOCAB = 49152
PHASE1_STEPS = 5000
PHASE2_STEPS = 20000
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS
LR = 6e-4
LR_MIN = 6e-5
WARMUP = 500
DENSITY = 0.20
CKPT_DIR = "checkpoints"
PHASE1_CKPT = f"{CKPT_DIR}/bench_phase1_d{D_MODEL}L{N_LAYERS}.pt"


def get_data_loader():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    buffer = []
    for sample in ds:
        tokens = tokenizer(sample["text"], truncation=False, add_special_tokens=False)["input_ids"]
        buffer.extend(tokens)
        while len(buffer) >= BATCH_SIZE * (SEQ_LEN + 1):
            batch_tokens = buffer[:BATCH_SIZE * (SEQ_LEN + 1)]
            buffer = buffer[BATCH_SIZE * (SEQ_LEN + 1):]
            t = torch.tensor(batch_tokens, dtype=torch.long).reshape(BATCH_SIZE, SEQ_LEN + 1)
            yield t[:, :SEQ_LEN], t[:, 1:SEQ_LEN + 1]


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, device, n_batches=20):
    model.eval()
    loader = get_data_loader()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        losses.append(out["loss"].item())
    val_loss = sum(losses) / len(losses) if losses else 20.0
    return val_loss, math.exp(min(val_loss, 20))


# ---------------------------------------------------------------------------
# bf16 baseline (from scratch, no ternary)
# ---------------------------------------------------------------------------

def run_bf16(device):
    """Standard bf16 training — the quality ceiling."""
    from model import TernaryTransformer, TernaryConfig
    from bitlinear import BitLinear, BitEmbedding

    # Monkey-patch to disable ternary
    _orig_bl = BitLinear.forward
    _orig_be = BitEmbedding.forward
    BitLinear.forward = lambda self, x: F.linear(x, self.weight, self.bias)
    BitEmbedding.forward = lambda self, ids: F.embedding(ids, self.weight)

    d_ff = ((int(D_MODEL * 2.69) + 7) // 8) * 8
    cfg = TernaryConfig(d_model=D_MODEL, n_heads=max(1, D_MODEL // 64),
                        d_ff=d_ff, n_layers=N_LAYERS, vocab_size=VOCAB,
                        max_seq_len=1024, group_size=min(128, D_MODEL))
    model = TernaryTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  bf16: {n_params/1e6:.2f}M params, full density, 25k steps")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                  weight_decay=0.1, eps=1e-8)

    wandb.init(project=WANDB_PROJECT, name=f"bench_bf16_d{D_MODEL}L{N_LAYERS}_25k",
               settings=wandb.Settings(init_timeout=180),
               config={"arm": "bf16", "steps": TOTAL_STEPS, "batch_size": BATCH_SIZE,
                       "d_model": D_MODEL, "n_layers": N_LAYERS, "density": 1.0})

    loader = get_data_loader()
    model.train()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > TOTAL_STEPS:
            break
        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, WARMUP, TOTAL_STEPS, LR, LR_MIN)
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

        if step % 500 == 0:
            avg = running_loss / 500
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - t_start
            print(f"    [bf16] step {step:>5d}/{TOTAL_STEPS} | loss {avg:.4f} | "
                  f"ppl {ppl:.1f} | {step/elapsed:.0f} step/s", flush=True)
            wandb.log({"train/loss": avg, "train/ppl": ppl, "lr": lr}, step=step)
            running_loss = 0.0

        if step % 2500 == 0:
            val_loss, val_ppl = evaluate(model, device)
            print(f"    [bf16] EVAL step {step}: val={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)
            model.train()

    val_loss, val_ppl = evaluate(model, device)
    elapsed = time.time() - t_start
    print(f"    [bf16] FINAL: val={val_loss:.4f} ppl={val_ppl:.1f} ({elapsed:.0f}s)")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=TOTAL_STEPS)
    wandb.finish()

    # Restore
    BitLinear.forward = _orig_bl
    BitEmbedding.forward = _orig_be
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Shared Phase 1
# ---------------------------------------------------------------------------

def ensure_phase1(device):
    """Train Phase 1 if checkpoint doesn't exist."""
    from model import TernaryTransformer, TernaryConfig

    if os.path.exists(PHASE1_CKPT):
        print(f"\n  Phase 1 checkpoint exists: {PHASE1_CKPT}")
        ckpt = torch.load(PHASE1_CKPT, map_location=device, weights_only=False)
        return ckpt["config"]

    d_ff = ((int(D_MODEL * 2.69) + 7) // 8) * 8
    cfg = TernaryConfig(d_model=D_MODEL, n_heads=max(1, D_MODEL // 64),
                        d_ff=d_ff, n_layers=N_LAYERS, vocab_size=VOCAB,
                        max_seq_len=1024, group_size=min(128, D_MODEL))
    model = TernaryTransformer(cfg).to(device)
    print(f"\n  Phase 1: training STE from scratch ({PHASE1_STEPS} steps)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                  weight_decay=0.1, eps=1e-8)
    loader = get_data_loader()
    model.train()
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE1_STEPS:
            break
        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, WARMUP, PHASE1_STEPS, LR, LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, labels=y)["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0:
            print(f"    [P1] step {step}/{PHASE1_STEPS} | loss {loss.item():.4f}", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, PHASE1_CKPT)
    elapsed = time.time() - t_start
    print(f"    Phase 1 done: {elapsed:.0f}s, saved to {PHASE1_CKPT}")
    return cfg


def load_and_prune(cfg, device):
    """Load Phase 1 and prune to target density."""
    from model import TernaryTransformer
    from run_ab_eggroll import perlayer_prune

    ckpt = torch.load(PHASE1_CKPT, map_location=device, weights_only=False)
    model = TernaryTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    pruned_data, actual_density = perlayer_prune(model, DENSITY, device, skip_embed_lmhead=True)
    return model, pruned_data, actual_density


# ---------------------------------------------------------------------------
# STE+Adam Phase 2
# ---------------------------------------------------------------------------

def run_ste(cfg, device):
    """STE+Adam continuation after pruning — standard ternary training."""
    from model import TernaryTransformer

    model, pruned_data, actual_density = load_and_prune(cfg, device)
    print(f"\n  STE+Adam: Phase 2, {PHASE2_STEPS} steps at {actual_density*100:.0f}% density")

    val_loss0, val_ppl0 = evaluate(model, device)
    print(f"    Post-prune: val={val_loss0:.4f} ppl={val_ppl0:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                  weight_decay=0.1, eps=1e-8)

    wandb.init(project=WANDB_PROJECT, name=f"bench_ste_d{D_MODEL}L{N_LAYERS}_25k",
               settings=wandb.Settings(init_timeout=180),
               config={"arm": "ste", "steps": PHASE2_STEPS, "batch_size": BATCH_SIZE,
                       "d_model": D_MODEL, "n_layers": N_LAYERS, "density": actual_density,
                       "phase1_steps": PHASE1_STEPS})
    wandb.log({"val/loss": val_loss0, "val/ppl": val_ppl0}, step=0)

    loader = get_data_loader()
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
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - t_start
            print(f"    [ste] step {step:>5d}/{PHASE2_STEPS} | loss {avg:.4f} | "
                  f"ppl {ppl:.1f} | {step/elapsed:.0f} step/s", flush=True)
            wandb.log({"train/loss": avg, "train/ppl": ppl, "lr": lr}, step=step)
            running_loss = 0.0

        if step % 2500 == 0:
            val_loss, val_ppl = evaluate(model, device)
            print(f"    [ste] EVAL step {step}: val={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)
            model.train()

    val_loss, val_ppl = evaluate(model, device)
    elapsed = time.time() - t_start
    print(f"    [ste] FINAL: val={val_loss:.4f} ppl={val_ppl:.1f} ({elapsed:.0f}s)")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=PHASE2_STEPS)
    wandb.finish()
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Pressure Flip Phase 2
# ---------------------------------------------------------------------------

def run_pf(cfg, device):
    """Pressure Flip continuation after pruning — 2 bytes/param."""
    from pressure_flip import create_pressure_flip_model, PressureFlipTrainer

    model, pruned_data, actual_density = load_and_prune(cfg, device)
    print(f"\n  PressureFlip: Phase 2, {PHASE2_STEPS} steps at {actual_density*100:.0f}% density")

    pf_model = create_pressure_flip_model(model, pruned_data, cfg, device)
    del model

    val_loss0, val_ppl0 = evaluate(pf_model, device)
    print(f"    Post-build: val={val_loss0:.4f} ppl={val_ppl0:.1f}")

    trainer = PressureFlipTrainer(
        pf_model,
        threshold_start=10,
        threshold_end=40,
        threshold_schedule="cosine",
        scale_lr=0.0,
        total_steps=PHASE2_STEPS,
    )

    wandb.init(project=WANDB_PROJECT, name=f"bench_pf_d{D_MODEL}L{N_LAYERS}_25k",
               settings=wandb.Settings(init_timeout=180),
               config={"arm": "pressure_flip", "steps": PHASE2_STEPS,
                       "batch_size": BATCH_SIZE, "d_model": D_MODEL,
                       "n_layers": N_LAYERS, "density": actual_density,
                       "phase1_steps": PHASE1_STEPS,
                       "threshold_start": 10, "threshold_end": 40})
    wandb.log({"val/loss": val_loss0, "val/ppl": val_ppl0}, step=0)

    loader = get_data_loader()
    running_loss = 0.0
    running_flips = 0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break

        stats = trainer.step(x, y, device)
        running_loss += stats["loss"]
        running_flips += stats["n_flipped"]

        if step % 500 == 0:
            avg = running_loss / 500
            avg_flips = running_flips / 500
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - t_start
            print(f"    [pf] step {step:>5d}/{PHASE2_STEPS} | loss {avg:.4f} | "
                  f"ppl {ppl:.1f} | {step/elapsed:.0f} step/s | "
                  f"flips {avg_flips:.0f}/step | thresh={stats['threshold']}",
                  flush=True)
            wandb.log({"train/loss": avg, "train/ppl": ppl,
                       "pf/flips_per_step": avg_flips,
                       "pf/threshold": stats["threshold"],
                       "pf/mean_pressure": stats["mean_pressure"]}, step=step)
            running_loss = 0.0
            running_flips = 0

        if step % 2500 == 0:
            pf_model.eval()
            val_loss, val_ppl = evaluate(pf_model, device)
            print(f"    [pf] EVAL step {step}: val={val_loss:.4f} ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

    pf_model.eval()
    val_loss, val_ppl = evaluate(pf_model, device)
    elapsed = time.time() - t_start
    print(f"    [pf] FINAL: val={val_loss:.4f} ppl={val_ppl:.1f} ({elapsed:.0f}s)")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=PHASE2_STEPS)
    wandb.finish()
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["bf16", "ste", "pf", "all"], default="all")
    args = parser.parse_args()
    device = torch.device("cuda")

    print(f"{'='*60}")
    print(f"  BENCHMARK: bf16 vs STE vs PressureFlip")
    print(f"  d={D_MODEL}, L={N_LAYERS}, B={BATCH_SIZE}, seq={SEQ_LEN}")
    print(f"  Phase 1: {PHASE1_STEPS} steps (STE, full density)")
    print(f"  Phase 2: {PHASE2_STEPS} steps (method-specific, {DENSITY*100:.0f}% density)")
    print(f"  bf16: {TOTAL_STEPS} steps from scratch (no ternary)")
    print(f"{'='*60}")

    results = {}

    if args.arm in ("bf16", "all"):
        val, ppl = run_bf16(device)
        results["bf16"] = (val, ppl)
        torch.cuda.empty_cache()

    if args.arm in ("ste", "pf", "all"):
        cfg = ensure_phase1(device)
        torch.cuda.empty_cache()

    if args.arm in ("ste", "all"):
        val, ppl = run_ste(cfg, device)
        results["ste"] = (val, ppl)
        torch.cuda.empty_cache()

    if args.arm in ("pf", "all"):
        val, ppl = run_pf(cfg, device)
        results["pf"] = (val, ppl)
        torch.cuda.empty_cache()

    if results:
        print(f"\n{'='*60}")
        print(f"  RESULTS (25k equivalent steps)")
        print(f"  {'Method':<20} {'Val Loss':<10} {'Val PPL':<10} {'Mem/param'}")
        print(f"  {'-'*55}")
        for name, (val, ppl) in results.items():
            mem = {"bf16": "14 bytes", "ste": "14 bytes", "pf": "2 bytes"}[name]
            print(f"  {name:<20} {val:<10.4f} {ppl:<10.1f} {mem}")
        print(f"{'='*60}")
