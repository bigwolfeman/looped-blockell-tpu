"""Training script for ternary vs baseline A/B comparison.

Usage:
  python train.py --mode ternary --steps 20000 --name ternary_v1
  python train.py --mode baseline --steps 20000 --name baseline_v1
  python train.py --mode flip    --steps 20000 --name flip_v1

Three modes: baseline (bf16), ternary (STE+shadow), flip (no shadow, discrete)
"""

import argparse
import time
import math

import torch
import torch.nn.functional as F
import wandb

from model import TernaryTransformer, TernaryConfig
from baseline import BaselineTransformer, BaselineConfig
from flip_model import FlipTransformer, FlipTransformerConfig
from cms_flip_model import CMSFlipTransformer, CMSFlipConfig
from sprt_flip_model import SPRTFlipTransformer, SPRTFlipConfig
from dual_sprt_model import DualSPRTTransformer, DualSPRTConfig
from dqt_model import DQTTransformer, DQTConfig
from ecpg_model import ECPGTransformer, ECPGConfig
from shadow_drop_model import ShadowDropTransformer, ShadowDropConfig
from qsgd_model import QSGDTransformer, QSGDConfig
from lste_model import LSTETransformer, LSTEConfig


def get_data_loader(batch_size: int, seq_len: int, buffer_tokens: int = 50_000_000):
    """Streaming OpenWebText loader (same as main pipeline)."""
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


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float = 0.0) -> float:
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train(args):
    device = torch.device("cuda")

    # Build model
    is_flip = args.mode in ("flip", "cms_flip", "sprt", "dual_sprt", "dqt", "ecpg", "shadow_drop", "qsgd", "lste")

    if args.mode == "ternary":
        cfg = TernaryConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
        )
        model = TernaryTransformer(cfg).to(device)
    elif args.mode == "flip":
        cfg = FlipTransformerConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size, flip_threshold=args.flip_threshold,
        )
        model = FlipTransformer(cfg).to(device)
    elif args.mode == "cms_flip":
        cfg = CMSFlipConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
            tau_base=args.tau_base, tau_structural=args.tau_structural,
        )
        model = CMSFlipTransformer(cfg).to(device)
    elif args.mode == "sprt":
        cfg = SPRTFlipConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
            p_signal=args.p_signal, alpha=args.alpha_flip,
        )
        model = SPRTFlipTransformer(cfg).to(device)
    elif args.mode == "dual_sprt":
        cfg = DualSPRTConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
            p_sign=args.p_signal, p_mag=args.p_signal,
            alpha_sign=args.alpha_flip,
        )
        model = DualSPRTTransformer(cfg).to(device)
    elif args.mode == "dqt":
        cfg = DQTConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
            flip_rate=args.flip_rate, error_decay=args.error_decay,
        )
        model = DQTTransformer(cfg).to(device)
    elif args.mode == "ecpg":
        cfg = ECPGConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
            base_rate=args.base_rate,
            max_group_flips=args.max_group_flips,
        )
        model = ECPGTransformer(cfg).to(device)
    elif args.mode == "lste":
        cfg = LSTEConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
        )
        model = LSTETransformer(cfg).to(device)
    elif args.mode == "qsgd":
        cfg = QSGDConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size, momentum=args.sgd_momentum,
        )
        model = QSGDTransformer(cfg).to(device)
    elif args.mode == "shadow_drop":
        cfg = ShadowDropConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
            group_size=args.group_size,
            flip_rate=args.flip_rate, error_decay=args.error_decay,
            drop_step=args.drop_step,
        )
        model = ShadowDropTransformer(cfg).to(device)
    else:
        cfg = BaselineConfig(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
        )
        model = BaselineTransformer(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # Optimizer: for flip mode, only optimize non-flip params (norms, residual alpha)
    # Exception: shadow_drop starts with ALL params optimized (phase 1)
    if is_flip and args.mode != "shadow_drop":
        flip_param_ids = {id(p) for p in model.get_flip_params()}
        opt_params = [p for p in model.parameters()
                      if id(p) not in flip_param_ids and p.requires_grad]
        n_opt = sum(p.numel() for p in opt_params)
        print(f"  Optimizer covers {n_opt/1e6:.2f}M params "
              f"(norms + residual alpha, NOT weights)")
    else:
        opt_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        opt_params, lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, eps=1e-8,
    )

    # wandb
    wandb.init(
        project="bonsai-ternary-test",
        name=args.name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "mode": args.mode,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "n_heads": args.n_heads,
            "n_params": n_params,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "group_size": args.group_size,
        },
    )

    # Data
    loader = get_data_loader(args.batch_size, 1024)

    # Training loop
    model.train()
    step = 0
    t_start = time.time()
    running_loss = 0.0

    for x, y in loader:
        if step >= args.steps:
            break

        x, y = x.to(device), y.to(device)

        # LR schedule
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward (autocast for activations, but ternary handles its own quantization)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
            loss = out["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Flip step (discrete weight updates) — must be before optimizer.step()
        flip_stats = {"n_flips": 0, "n_sign": 0, "n_structural": 0}
        if is_flip:
            # Temperature schedule: cosine anneal from 1.0 → 0.1
            temperature = 0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2
            if args.mode in ("qsgd", "lste"):
                result = model.flip_step(scale_lr=args.scale_lr, lr=lr)
            elif args.mode in ("dqt", "ecpg", "shadow_drop"):
                result = model.flip_step(scale_lr=args.scale_lr, temperature=temperature)
            else:
                result = model.flip_step(scale_lr=args.scale_lr)
            if isinstance(result, dict):
                flip_stats = result
            else:
                flip_stats["n_flips"] = result

        optimizer.step()

        # Shadow drop transition: rebuild optimizer excluding weight params
        if args.mode == "shadow_drop" and hasattr(model, 'maybe_drop_shadows'):
            if model.maybe_drop_shadows(step):
                # Rebuild optimizer with only non-weight params
                flip_param_ids = {id(p) for p in model.get_flip_params()}
                opt_params = [p for p in model.parameters()
                              if id(p) not in flip_param_ids and p.requires_grad]
                n_opt = sum(p.numel() for p in opt_params)
                optimizer = torch.optim.AdamW(
                    opt_params, lr=args.lr, betas=(0.9, 0.95),
                    weight_decay=args.weight_decay, eps=1e-8,
                )
                print(f"  Rebuilt optimizer: {n_opt/1e6:.2f}M params (norms only)")

        # Logging
        running_loss += loss.item()
        step += 1

        if step % 20 == 0:
            avg_loss = running_loss / 20
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = step / elapsed
            vram = torch.cuda.memory_allocated() / 1e9

            print(f"  step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:7.1f} | "
                  f"lr {lr:.2e} | {steps_per_s:.1f} step/s | {vram:.1f}GB")

            log_data = {
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "perf/vram_gb": vram,
                "lr": lr,
            }

            if is_flip:
                log_data["flip/n_flips"] = flip_stats["n_flips"]
                log_data["flip/n_sign"] = flip_stats.get("n_sign", 0)
                log_data["flip/n_structural"] = flip_stats.get("n_structural", 0)
                log_data["flip/temperature"] = temperature
                stats = model.ternary_stats()
                log_data["ternary/neg_frac"] = stats["neg_frac"]
                log_data["ternary/zero_frac"] = stats["zero_frac"]
                log_data["ternary/pos_frac"] = stats["pos_frac"]
                if "avg_flip_count" in stats:
                    log_data["flip/avg_flip_count"] = stats["avg_flip_count"]
                    log_data["flip/max_flip_count"] = stats["max_flip_count"]
                if "avg_error" in stats:
                    log_data["error/avg_abs"] = stats["avg_error"]
                    log_data["error/max_abs"] = stats["max_error"]
                if "avg_ste_gain" in stats:
                    log_data["lste/avg_gain"] = stats["avg_ste_gain"]
                    log_data["lste/avg_temp"] = stats["avg_ste_temp"]

            wandb.log(log_data, step=step)
            running_loss = 0.0

        if step % 500 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(loader):
                    if i >= 10:
                        break
                    # Use smaller seq for eval to avoid OOM on logits
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
    print(f"\nDone. {step} steps in {time.time() - t_start:.0f}s")

    # Save checkpoint
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, f"checkpoints/{args.name}.pt")
    print(f"Saved: checkpoints/{args.name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ternary", "baseline", "flip", "cms_flip", "sprt", "dual_sprt", "dqt", "ecpg", "shadow_drop", "qsgd", "lste"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1376)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--flip_threshold", type=float, default=10.0)
    parser.add_argument("--scale_lr", type=float, default=1e-3)
    parser.add_argument("--tau_base", type=float, default=0.5)
    parser.add_argument("--tau_structural", type=float, default=1.5)
    parser.add_argument("--p_signal", type=float, default=0.6)
    parser.add_argument("--alpha_flip", type=float, default=0.01)
    parser.add_argument("--flip_rate", type=float, default=0.01)
    parser.add_argument("--error_decay", type=float, default=0.99)
    parser.add_argument("--base_rate", type=float, default=0.005)
    parser.add_argument("--max_group_flips", type=float, default=0.05)
    parser.add_argument("--drop_step", type=int, default=5000)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    args = parser.parse_args()

    import os
    os.makedirs("checkpoints", exist_ok=True)
    train(args)
