"""Training loop for subquadratic attention ablation.

Usage:
  python train.py --attn dense --name dense_baseline --seq_len 4096 --steps 20000
  python train.py --attn sliding_window --name sw_4k --seq_len 4096 --steps 20000
"""

import argparse
import math
import os
import time

import torch
import wandb

from model import SubQTransformer, ModelConfig


def get_data_loader(batch_size: int, seq_len: int):
    """Streaming OpenWebText loader."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    buffer = []
    for sample in ds:
        tokens = tokenizer(
            sample["text"], truncation=False, add_special_tokens=False
        )["input_ids"]
        buffer.extend(tokens)

        while len(buffer) >= batch_size * (seq_len + 1):
            batch_tokens = buffer[: batch_size * (seq_len + 1)]
            buffer = buffer[batch_size * (seq_len + 1) :]
            t = torch.tensor(batch_tokens, dtype=torch.long).reshape(
                batch_size, seq_len + 1
            )
            yield t[:, :seq_len], t[:, 1 : seq_len + 1]


def cosine_lr(step: int, warmup: int, total: int, lr_max: float) -> float:
    lr_min = lr_max * 0.1
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, n_batches: int = 10) -> tuple[float, float]:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        losses.append(out["loss"].item())
    model.train()
    avg = sum(losses) / len(losses) if losses else float("inf")
    return avg, math.exp(min(avg, 20))


def measure_scaling(model, device, seq_lengths, batch_size=2, n_iters=5):
    """Measure wall-clock and VRAM at each sequence length."""
    results = {}
    model.eval()
    for sl in seq_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 1000, (batch_size, sl), device=device)

        # warmup
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        avg_ms = sum(times) / len(times) * 1000
        results[sl] = {"time_ms": avg_ms, "peak_vram_mb": peak_mb}
        print(f"  seq_len={sl:6d}: {avg_ms:8.1f}ms  {peak_mb:8.0f}MB")

    model.train()
    return results


def train(args):
    device = torch.device("cuda")

    attn_kwargs = {}
    if args.window_size:
        attn_kwargs["window_size"] = args.window_size

    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab_size=49152,
        max_seq_len=args.seq_len,
        attn_type=args.attn,
        attn_kwargs=attn_kwargs,
    )
    model = SubQTransformer(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    wandb.init(
        project="subq-attention-ablation",
        name=args.name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "attn_type": args.attn,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "steps": args.steps,
            "n_params": sum(p.numel() for p in model.parameters()),
            **attn_kwargs,
        },
    )

    # Measure scaling before training
    if args.measure_scaling:
        print("\n=== Scaling measurement (untrained) ===")
        lengths = [2048, 4096, 8192, 16384]
        if args.seq_len >= 32768:
            lengths.append(32768)
        scaling = measure_scaling(model, device, lengths)
        for sl, m in scaling.items():
            wandb.log({
                f"scaling/time_ms_{sl}": m["time_ms"],
                f"scaling/vram_mb_{sl}": m["peak_vram_mb"],
            }, step=0)

    loader = get_data_loader(args.batch_size, args.seq_len)

    model.train()
    step = 0
    t_start = time.time()
    running_loss = 0.0

    for x, y in loader:
        if step >= args.steps:
            break

        x, y = x.to(device), y.to(device)
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
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
            vram = torch.cuda.memory_allocated() / 1e9
            print(
                f"  step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:7.1f} | "
                f"lr {lr:.2e} | {step/elapsed:.1f} step/s | {vram:.1f}GB"
            )
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/ppl": ppl,
                    "perf/step_per_s": step / elapsed,
                    "perf/vram_gb": vram,
                    "lr": lr,
                },
                step=step,
            )
            running_loss = 0.0

        if step % 500 == 0:
            val_loss, val_ppl = evaluate(model, loader, device)
            print(f"  EVAL step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

    # Final scaling measurement
    if args.measure_scaling:
        print("\n=== Scaling measurement (trained) ===")
        lengths = [2048, 4096, 8192, 16384]
        if args.seq_len >= 32768:
            lengths.append(32768)
        scaling = measure_scaling(model, device, lengths)
        for sl, m in scaling.items():
            wandb.log({
                f"scaling_final/time_ms_{sl}": m["time_ms"],
                f"scaling_final/vram_mb_{sl}": m["peak_vram_mb"],
            }, step=step)

    wandb.finish()

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{args.name}.pt"
    torch.save(
        {"step": step, "model_state_dict": model.state_dict(), "config": cfg},
        ckpt_path,
    )
    print(f"\nDone. {step} steps in {time.time() - t_start:.0f}s → {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", type=str, required=True, help="Attention type from registry")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--measure_scaling", action="store_true")
    args = parser.parse_args()
    train(args)
