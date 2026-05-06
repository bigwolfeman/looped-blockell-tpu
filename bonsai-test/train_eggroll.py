"""EGGROLL training script — pure evolution strategies, zero backprop."""

import argparse
import time
import math

import torch
import wandb

from eggroll_model import EggrollTransformer, EggrollConfig


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


def train(args):
    device = torch.device("cuda")

    cfg = EggrollConfig(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
        group_size=args.group_size,
        pop_size=args.pop_size, sigma=args.sigma, es_lr=args.es_lr,
    )
    model = EggrollTransformer(cfg).to(device)

    n_params = sum(m.ternary.numel() for _, m, _ in model._es_modules)

    wandb.init(
        project="bonsai-ternary-test",
        name=args.name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "mode": "eggroll",
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_params": n_params,
            "pop_size": args.pop_size,
            "sigma": args.sigma,
            "es_lr": args.es_lr,
            "batch_size": args.batch_size,
            "steps": args.steps,
        },
    )

    loader = get_data_loader(args.batch_size, 1024)

    model.eval()  # no dropout, no training-specific behavior
    step = 0
    t_start = time.time()
    running_loss = 0.0

    for x, y in loader:
        if step >= args.steps:
            break

        x, y = x.to(device), y.to(device)

        # ES step — no gradients, no backward
        with torch.no_grad():
            # Evaluate base model (unperturbed) for logging
            base_out = model(x, labels=y)
            base_loss = base_out["loss"].item()

            # Evolution step
            flip_stats = model.es_step(x, y, step=step)

        running_loss += base_loss
        step += 1

        if step % 20 == 0:
            avg_loss = running_loss / 20
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = step / elapsed
            vram = torch.cuda.memory_allocated() / 1e9

            print(f"  step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:7.1f} | "
                  f"{steps_per_s:.2f} step/s | {vram:.1f}GB | "
                  f"flips {flip_stats['n_flips']}")

            log_data = {
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "perf/vram_gb": vram,
                "flip/n_flips": flip_stats["n_flips"],
                "flip/n_sign": flip_stats.get("n_sign", 0),
                "flip/n_structural": flip_stats.get("n_structural", 0),
            }

            stats = model.ternary_stats()
            log_data["ternary/neg_frac"] = stats["neg_frac"]
            log_data["ternary/zero_frac"] = stats["zero_frac"]
            log_data["ternary/pos_frac"] = stats["pos_frac"]

            wandb.log(log_data, step=step)
            running_loss = 0.0

        if step % 500 == 0:
            val_losses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(loader):
                    if i >= 10:
                        break
                    vx = vx[:, :512].to(device)
                    vy = vy[:, :512].to(device)
                    vout = model(vx, labels=vy)
                    val_losses.append(vout["loss"].item())
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  EVAL step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

    wandb.finish()
    print(f"\nDone. {step} steps in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1376)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--es_lr", type=float, default=0.01)
    args = parser.parse_args()

    train(args)
