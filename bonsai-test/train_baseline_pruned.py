"""Baseline: BPTT → Prune → Continue STE (no EGGROLL).

A/B comparison partner for train_pipeline.py.
Loads Phase 1 checkpoint, prunes to target density, continues STE+Adam.
This tests whether EGGROLL adds anything beyond continued backprop.
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import wandb

from model import TernaryTransformer, TernaryConfig
from bitlinear import BitLinear, BitEmbedding


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
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def prune_model(model, density):
    """Prune by shadow weight magnitude to target density."""
    all_mags = []
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, (BitLinear, BitEmbedding)):
            all_mags.append(mod.weight.data.abs().reshape(-1))
            layers.append((name, mod))
    all_mags = torch.cat(all_mags)

    k = int(density * all_mags.numel())
    threshold = torch.kthvalue(all_mags, all_mags.numel() - k).values.item()

    total_alive = 0
    total_params = 0
    for name, mod in layers:
        mask = mod.weight.data.abs() >= threshold
        mod.weight.data *= mask.float()
        alive = mask.sum().item()
        total = mask.numel()
        total_alive += alive
        total_params += total
        print(f"  Pruned {name}: {alive}/{total} alive ({alive/total*100:.1f}%)")

    print(f"  Overall: {total_alive}/{total_params} ({total_alive/total_params*100:.1f}%)")
    return total_alive, total_params


def train(args):
    device = torch.device("cuda")

    # Load Phase 1 checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("Run train_pipeline.py first to generate Phase 1 checkpoint")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TernaryTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    phase1_step = ckpt["step"]
    print(f"  Loaded at step {phase1_step}")

    # Prune
    print(f"\n  PRUNING to {args.prune_density*100:.0f}% density")
    n_alive, n_total = prune_model(model, args.prune_density)

    # Continue STE+Adam training
    total_steps = phase1_step + args.continue_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-8,
    )

    wandb.init(
        project="bonsai-ternary-test",
        name=args.name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "mode": "baseline_pruned_ste",
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "phase1_steps": phase1_step,
            "continue_steps": args.continue_steps,
            "prune_density": args.prune_density,
            "n_alive": n_alive,
            "n_total": n_total,
        },
    )

    loader = get_data_loader(args.batch_size, 1024)
    model.train()
    step = phase1_step
    t_start = time.time()
    running_loss = 0.0

    for x, y in loader:
        if step >= total_steps:
            break

        x, y = x.to(device), y.to(device)

        lr = cosine_lr(step, 500, total_steps, args.lr, args.lr * 0.1)
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
        step += 1

        if step % 20 == 0:
            avg_loss = running_loss / 20
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_done = step - phase1_step
            steps_per_s = steps_done / elapsed if elapsed > 0 else 0
            vram = torch.cuda.memory_allocated() / 1e9

            print(f"  step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:7.1f} | "
                  f"lr {lr:.2e} | {steps_per_s:.1f} step/s | {vram:.1f}GB")

            wandb.log({
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "perf/vram_gb": vram,
                "lr": lr,
                "phase": 3,
            }, step=step)
            running_loss = 0.0

        if step % 500 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(loader):
                    if i >= 10:
                        break
                    vx = vx[:, :512].to(device)
                    vy = vy[:, :512].to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vout = model(vx, labels=vy)
                    val_losses.append(vout["loss"].item())
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  EVAL step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)
            model.train()

    wandb.finish()
    print(f"\nDone. {step - phase1_step} steps in {time.time() - t_start:.0f}s")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, f"checkpoints/{args.name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to Phase 1 checkpoint")
    parser.add_argument("--continue_steps", type=int, default=5000)
    parser.add_argument("--prune_density", type=float, default=0.20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    args = parser.parse_args()

    train(args)
