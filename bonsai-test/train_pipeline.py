"""BPTT → Prune → EGGROLL three-phase training pipeline.

Phase 1: STE+Adam on TernaryTransformer (structure discovery via backprop)
Phase 2: Score-based weight pruning, create binary masks at target density
Phase 3: EGGROLL evolution strategies on pruned EggrollTransformer (no backprop)

Single wandb run spanning all three phases with phase-transition markers.

Usage:
  python train_pipeline.py --name pipeline_v1 \
      --phase1_steps 20000 --prune_density 0.20 \
      --phase3_steps 5000 --pop_size 4096 --sigma 0.01 --es_lr 0.05 \
      --batch_size 4 --d_model 512
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
from bitlinear import ternary_quantize


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_data_loader(batch_size: int, seq_len: int):
    """Streaming OpenWebText loader (matches train.py exactly, no trust_remote_code)."""
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
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def eval_model(model, loader, device, n_batches: int = 10, seq_len: int = 512) -> tuple[float, float]:
    """Evaluate model on n_batches, return (val_loss, val_ppl)."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, (vx, vy) in enumerate(loader):
            if i >= n_batches:
                break
            vx = vx[:, :seq_len].to(device)
            vy = vy[:, :seq_len].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vout = model(vx, labels=vy)
            val_losses.append(vout["loss"].item())
    val_loss = sum(val_losses) / max(1, len(val_losses))
    val_ppl = math.exp(min(val_loss, 20))
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Phase 1: BPTT with STE+Adam
# ---------------------------------------------------------------------------

def phase1_train(args, loader, device, global_step_offset: int = 0) -> tuple[TernaryTransformer, TernaryConfig]:
    print(f"\n{'='*60}")
    print(f"  PHASE 1: BPTT with STE+Adam  ({args.phase1_steps} steps)")
    print(f"{'='*60}")
    sys.stdout.flush()

    cfg = TernaryConfig(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
        group_size=args.group_size,
    )
    model = TernaryTransformer(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay, eps=1e-8,
    )

    wandb.log({"phase": 1}, step=global_step_offset)

    model.train()
    running_loss = 0.0
    t_start = time.time()

    for step_local, (x, y) in enumerate(loader):
        if step_local >= args.phase1_steps:
            break

        global_step = global_step_offset + step_local
        x, y = x.to(device), y.to(device)

        lr = cosine_lr(step_local, args.warmup, args.phase1_steps, args.lr, args.lr * 0.1)
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

        if (step_local + 1) % 20 == 0:
            avg_loss = running_loss / 20
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = (step_local + 1) / elapsed
            vram = torch.cuda.memory_allocated() / 1e9

            print(f"  [P1] step {step_local+1:6d}/{args.phase1_steps} | "
                  f"loss {avg_loss:.4f} | ppl {ppl:7.1f} | lr {lr:.2e} | "
                  f"{steps_per_s:.1f} step/s | {vram:.1f}GB", flush=True)

            wandb.log({
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "perf/vram_gb": vram,
                "lr": lr,
                "phase": 1,
            }, step=global_step)
            running_loss = 0.0

        if (step_local + 1) % 500 == 0:
            val_loss, val_ppl = eval_model(model, loader, device)
            print(f"  [P1] EVAL step {step_local+1}: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "phase": 1}, step=global_step)
            model.train()

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{args.name}_phase1.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"  Phase 1 checkpoint saved: {ckpt_path}", flush=True)

    return model, cfg


# ---------------------------------------------------------------------------
# Phase 2: Prune
# ---------------------------------------------------------------------------

def phase2_prune(
    p1_model: TernaryTransformer,
    p1_cfg: TernaryConfig,
    prune_density: float,
    device: torch.device,
    global_step: int,
) -> dict:
    """Score weights by shadow-weight magnitude, keep top `prune_density` fraction.

    Returns a dict mapping module-name → {"ternary", "scales", "mask", "group_size"}
    that can be loaded into an EggrollTransformer.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Pruning to {prune_density*100:.0f}% density")
    print(f"{'='*60}")
    sys.stdout.flush()

    from bitlinear import BitLinear, BitEmbedding

    # Collect all ternary layers and their shadow weights
    layer_data = {}
    for name, mod in p1_model.named_modules():
        if isinstance(mod, (BitLinear, BitEmbedding)):
            shadow = mod.weight.detach().float()  # shadow weight in fp32
            layer_data[name] = {"shadow": shadow, "mod": mod}

    # Per-layer pruning: each layer pruned to target density independently
    # This prevents large layers (embedding, lm_head) from stealing budget
    n_total_all = sum(v["shadow"].numel() for v in layer_data.values())
    print(f"  Total weights: {n_total_all/1e6:.2f}M | Target density: {prune_density*100:.0f}%", flush=True)

    pruned_layers = {}
    total_alive = 0
    total_weights = 0

    for name, info in layer_data.items():
        shadow = info["shadow"]
        mod = info["mod"]
        group_size = mod.group_size

        # Per-layer threshold: keep top prune_density fraction by magnitude
        flat_mag = shadow.reshape(-1).abs()
        n_layer = flat_mag.numel()
        n_keep = max(1, int(n_layer * prune_density))
        if n_keep < n_layer:
            threshold = flat_mag.kthvalue(n_layer - n_keep).values.item()
        else:
            threshold = 0.0
        mask = (shadow.abs() >= threshold).to(torch.float32).to(device)

        # Quantize shadow weight to ternary (STE was doing this on-the-fly)
        # Shadow weight may not be exactly in [-1,0,1] scale — quantize properly
        try:
            w_q, scales = ternary_quantize(shadow, group_size)
        except Exception:
            # Fallback: find a valid group size
            total_el = shadow.numel()
            gs = group_size
            while total_el % gs != 0 and gs > 1:
                gs -= 1
            w_q, scales = ternary_quantize(shadow, gs)
            group_size = gs

        w_q = w_q.to(torch.int8).to(device)
        scales = scales.to(torch.bfloat16).to(device)

        # Zero out pruned positions in the ternary weight too
        w_q = (w_q.float() * mask).to(torch.int8)

        alive = mask.bool().sum().item()
        total_alive += alive
        total_weights += mask.numel()

        print(f"  {name}: {alive}/{mask.numel()} alive "
              f"({100.*alive/mask.numel():.1f}%)", flush=True)

        pruned_layers[name] = {
            "ternary": w_q,
            "scales": scales,
            "mask": mask,
            "group_size": group_size,
            "shape": tuple(shadow.shape),
        }

    actual_density = total_alive / max(1, total_weights)
    print(f"\n  Actual density: {actual_density*100:.2f}% ({total_alive/1e6:.2f}M alive weights)")

    wandb.log({
        "prune/density": actual_density,
        "prune/n_alive": total_alive,
        "prune/n_total": total_weights,
        "phase": 2,
    }, step=global_step)

    return pruned_layers


# ---------------------------------------------------------------------------
# Phase 3: EGGROLL on pruned model
# ---------------------------------------------------------------------------

def _transfer_weights(
    egg_model: EggrollTransformer,
    pruned_layers: dict,
    p1_model: TernaryTransformer,
    device: torch.device,
):
    """Copy ternary weights, scales, and masks from pruned Phase 1 data into EggrollTransformer.

    Also copies RMSNorm scale parameters and attention alpha buffers.
    """
    from bitlinear import BitLinear, BitEmbedding, RMSNorm as BitRMSNorm
    from eggroll_model import EggrollLinear, EggrollEmbedding
    from bitlinear import RMSNorm as BitRMSNormCls

    # Build lookup: module name -> EggrollLinear/EggrollEmbedding
    egg_modules: dict[str, EggrollLinear | EggrollEmbedding] = {}
    for name, mod in egg_model.named_modules():
        if isinstance(mod, (EggrollLinear, EggrollEmbedding)):
            egg_modules[name] = mod

    # Transfer ternary weights, scales, masks
    transferred = 0
    for name, data in pruned_layers.items():
        if name not in egg_modules:
            print(f"  [transfer] WARNING: {name} not found in EggrollTransformer — skipping")
            continue
        emod = egg_modules[name]

        # Shapes must match
        if emod.ternary.shape != data["ternary"].shape:
            print(f"  [transfer] WARNING: shape mismatch for {name}: "
                  f"{emod.ternary.shape} vs {data['ternary'].shape} — skipping")
            continue

        emod.ternary.copy_(data["ternary"])
        # scales may differ in dtype
        emod.scales.copy_(data["scales"].to(emod.scales.dtype))
        emod.weight_mask = data["mask"].to(device)
        emod.group_size = data["group_size"]
        transferred += 1

    print(f"  Transferred {transferred}/{len(pruned_layers)} ternary layers", flush=True)

    # Transfer RMSNorm scale parameters (they're regular nn.Parameters)
    p1_norms: dict[str, torch.Tensor] = {}
    for name, mod in p1_model.named_modules():
        if hasattr(mod, "scale") and isinstance(mod.scale, torch.nn.Parameter):
            p1_norms[name] = mod.scale.detach()

    egg_norms: dict[str, torch.nn.Parameter] = {}
    for name, mod in egg_model.named_modules():
        if hasattr(mod, "scale") and isinstance(mod.scale, torch.nn.Parameter):
            egg_norms[name] = mod.scale

    norm_count = 0
    for name, scale in p1_norms.items():
        if name in egg_norms and egg_norms[name].shape == scale.shape:
            egg_norms[name].data.copy_(scale.to(egg_norms[name].dtype))
            norm_count += 1
    print(f"  Transferred {norm_count} RMSNorm scales", flush=True)

    # Transfer attention alpha buffers (they live in EggrollAttention as buffers)
    p1_alphas: dict[str, torch.Tensor] = {}
    for name, mod in p1_model.named_modules():
        if hasattr(mod, "alpha") and mod.alpha is not None and isinstance(mod.alpha, torch.nn.Parameter):
            p1_alphas[name] = mod.alpha.detach()

    egg_alphas: dict[str, torch.Tensor] = {}
    for name, mod in egg_model.named_modules():
        if hasattr(mod, "alpha") and mod.alpha is not None and isinstance(mod.alpha, torch.Tensor):
            egg_alphas[name] = mod.alpha

    alpha_count = 0
    for name, alpha in p1_alphas.items():
        if name in egg_alphas and egg_alphas[name].shape == alpha.shape:
            egg_alphas[name].copy_(alpha.to(egg_alphas[name].dtype))
            alpha_count += 1
    print(f"  Transferred {alpha_count} attention alpha buffers", flush=True)


def phase3_eggroll(args, loader, device, pruned_layers: dict, p1_model: TernaryTransformer,
                   p1_cfg: TernaryConfig, global_step_offset: int = 0):
    print(f"\n{'='*60}")
    print(f"  PHASE 3: EGGROLL (ES, no backprop)  ({args.phase3_steps} steps)")
    print(f"  pop_size={args.pop_size}, sigma={args.sigma}, es_lr={args.es_lr}")
    print(f"{'='*60}")
    sys.stdout.flush()

    cfg = EggrollConfig(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_layers=args.n_layers, vocab_size=49152, max_seq_len=1024,
        group_size=args.group_size,
        pop_size=args.pop_size, sigma=args.sigma, es_lr=args.es_lr,
    )
    model = EggrollTransformer(cfg).to(device)

    # Transfer weights from pruned Phase 1 model
    _transfer_weights(model, pruned_layers, p1_model, device)

    n_alive = sum(
        (m.weight_mask.sum().item() if m.weight_mask is not None else m.ternary.numel())
        for _, m, _ in model._es_modules
    )
    n_total = sum(m.ternary.numel() for _, m, _ in model._es_modules)
    print(f"  ES params: {n_alive/1e6:.2f}M alive / {n_total/1e6:.2f}M total "
          f"({100.*n_alive/n_total:.1f}% density)", flush=True)

    wandb.log({"phase": 3, "es/n_alive": n_alive, "es/density": n_alive/n_total},
              step=global_step_offset)

    model.eval()  # No dropout, no training-specific behavior for ES
    running_loss = 0.0
    t_start = time.time()
    flip_stats_accum = {"n_flips": 0, "n_sign": 0, "n_structural": 0}

    for step_local, (x, y) in enumerate(loader):
        if step_local >= args.phase3_steps:
            break

        global_step = global_step_offset + step_local
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            # Base model loss for logging
            base_out = model(x, labels=y)
            base_loss = base_out["loss"].item()

            # ES step (no gradient)
            flip_stats = model.es_step(x, y, step=step_local)

        running_loss += base_loss
        for k in flip_stats_accum:
            flip_stats_accum[k] += flip_stats.get(k, 0)

        if (step_local + 1) % 20 == 0:
            avg_loss = running_loss / 20
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            steps_per_s = (step_local + 1) / elapsed
            vram = torch.cuda.memory_allocated() / 1e9

            print(f"  [P3] step {step_local+1:6d}/{args.phase3_steps} | "
                  f"loss {avg_loss:.4f} | ppl {ppl:7.1f} | "
                  f"{steps_per_s:.2f} step/s | {vram:.1f}GB | "
                  f"flips {flip_stats_accum['n_flips']}", flush=True)

            stats = model.ternary_stats()
            wandb.log({
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "perf/step_per_s": steps_per_s,
                "perf/vram_gb": vram,
                "flip/n_flips": flip_stats_accum["n_flips"],
                "flip/n_sign": flip_stats_accum["n_sign"],
                "flip/n_structural": flip_stats_accum["n_structural"],
                "ternary/neg_frac": stats["neg_frac"],
                "ternary/zero_frac": stats["zero_frac"],
                "ternary/pos_frac": stats["pos_frac"],
                "phase": 3,
            }, step=global_step)
            running_loss = 0.0
            flip_stats_accum = {"n_flips": 0, "n_sign": 0, "n_structural": 0}

        if (step_local + 1) % 500 == 0:
            val_loss, val_ppl = eval_model(model, loader, device)
            print(f"  [P3] EVAL step {step_local+1}: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}", flush=True)
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "phase": 3}, step=global_step)
            model.eval()  # es uses no_grad but restore eval mode

    # Final checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{args.name}_phase3.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"  Phase 3 checkpoint saved: {ckpt_path}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BPTT → Prune → EGGROLL pipeline")
    parser.add_argument("--name", required=True, help="Run name (used for wandb and checkpoints)")
    # Architecture
    parser.add_argument("--d_model",    type=int,   default=512)
    parser.add_argument("--n_heads",    type=int,   default=8)
    parser.add_argument("--d_ff",       type=int,   default=1376)
    parser.add_argument("--n_layers",   type=int,   default=6)
    parser.add_argument("--group_size", type=int,   default=128)
    # Phase 1
    parser.add_argument("--phase1_steps", type=int,   default=20000)
    parser.add_argument("--lr",           type=float, default=6e-4)
    parser.add_argument("--warmup",       type=int,   default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size",   type=int,   default=4,
                        help="Batch size for Phase 3 (EGGROLL). Phase 1 uses --phase1_batch_size")
    parser.add_argument("--phase1_batch_size", type=int, default=12,
                        help="Batch size for Phase 1 (STE+Adam, can be larger)")
    # Phase 2
    parser.add_argument("--prune_density", type=float, default=0.20)
    # Phase 3
    parser.add_argument("--phase3_steps", type=int,   default=5000)
    parser.add_argument("--pop_size",     type=int,   default=4096)
    parser.add_argument("--sigma",        type=float, default=0.01)
    parser.add_argument("--es_lr",        type=float, default=0.05)
    # Control
    parser.add_argument("--skip_phase1",  action="store_true",
                        help="Skip phase 1, load from checkpoints/<name>_phase1.pt")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Single wandb run for all phases
    wandb.init(
        project="bonsai-ternary-test",
        name=args.name,
        settings=wandb.Settings(init_timeout=180),
        config={
            "pipeline": "BPTT→Prune→EGGROLL",
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "group_size": args.group_size,
            "phase1_steps": args.phase1_steps,
            "phase3_steps": args.phase3_steps,
            "prune_density": args.prune_density,
            "pop_size": args.pop_size,
            "sigma": args.sigma,
            "es_lr": args.es_lr,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
    )

    print(f"\n{'='*60}")
    print(f"  EGGROLL PIPELINE: {args.name}")
    print(f"  d_model={args.d_model}, n_layers={args.n_layers}, d_ff={args.d_ff}")
    print(f"  Phase1: {args.phase1_steps} steps | Prune: {args.prune_density*100:.0f}% | Phase3: {args.phase3_steps} steps")
    print(f"{'='*60}", flush=True)

    # Separate loaders for Phase 1 (large batch) and Phase 3 (small batch for EGGROLL)
    loader_p1 = get_data_loader(args.phase1_batch_size, 1024)
    loader_p3 = get_data_loader(args.batch_size, 1024)

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    if args.skip_phase1:
        ckpt_path = f"checkpoints/{args.name}_phase1.pt"
        print(f"  Loading Phase 1 checkpoint from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        p1_cfg = ckpt["config"]
        p1_model = TernaryTransformer(p1_cfg).to(device)
        p1_model.load_state_dict(ckpt["model_state_dict"])
        global_step_after_p1 = args.phase1_steps  # logical step offset
    else:
        p1_model, p1_cfg = phase1_train(args, loader_p1, device, global_step_offset=0)
        global_step_after_p1 = args.phase1_steps

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    pruned_layers = phase2_prune(
        p1_model, p1_cfg, args.prune_density, device,
        global_step=global_step_after_p1,
    )

    # ------------------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------------------
    phase3_eggroll(
        args, loader_p3, device, pruned_layers, p1_model, p1_cfg,
        global_step_offset=global_step_after_p1 + 1,
    )

    wandb.finish()
    print("\nPipeline complete.", flush=True)


if __name__ == "__main__":
    main()
