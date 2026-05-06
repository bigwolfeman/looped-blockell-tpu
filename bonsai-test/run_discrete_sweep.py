"""Discrete ternary optimization sweep — Gumbel-Softmax, STG, REINMAX, Mirror Descent.

All share: d=128, L=3, B=8, seq=256, 20k Phase 2 steps.
Phase 1 checkpoint: checkpoints/bench_phase1_d128L3.pt

Usage:
  python run_discrete_sweep.py                    # run all at full density
  python run_discrete_sweep.py --density 0.2      # 20% density (pruned)
  python run_discrete_sweep.py --exp 1            # just Gumbel-Softmax
  python run_discrete_sweep.py --exp 0            # just STE+Adam bf16 baseline
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import wandb

torch.set_float32_matmul_precision("high")
os.environ["PYTHONUNBUFFERED"] = "1"

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
DENSITY = 1.0
PHASE1_CKPT = "checkpoints/bench_phase1_d128L3.pt"
LOG_EVERY = 50
EVAL_EVERY = 1000


def get_loader():
    from run_ab_eggroll import get_data_loader
    return get_data_loader(batch_size=B, seq_len=SEQ)


def full_density_data(model, device):
    """Create weight data with 100% density (no pruning)."""
    from bitlinear import BitLinear, BitEmbedding, ternary_quantize
    data = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, (BitLinear, BitEmbedding)):
            continue
        w = mod.weight.detach()
        gs = mod.group_size
        w_q, scales = ternary_quantize(w, gs)
        mask = torch.ones_like(w, device=device)
        data[name] = {
            "ternary": w_q.to(torch.int8).to(device),
            "scales": scales.to(torch.bfloat16).to(device),
            "mask": mask, "group_size": gs, "shape": tuple(w.shape),
        }
    return data


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def tau_anneal(step, total, tau_start, tau_end):
    """Exponential temperature annealing."""
    if tau_start <= tau_end:
        return tau_start
    rate = math.log(tau_start / tau_end) / total
    return max(tau_end, tau_start * math.exp(-rate * step))


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
    model.train()
    if not losses:
        return 20.0, math.exp(20)
    val = sum(losses) / len(losses)
    return val, math.exp(min(val, 20))


def load_phase1(device):
    from model import TernaryTransformer
    ckpt = torch.load(PHASE1_CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TernaryTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def get_weight_data(model, device):
    """Get ternary weight data — full density or pruned depending on DENSITY."""
    if DENSITY >= 1.0:
        return full_density_data(model, device), 1.0
    from run_ab_eggroll import perlayer_prune
    return perlayer_prune(model, DENSITY, device, skip_embed_lmhead=True)


def train_method(model, device, method_name, wandb_name, wandb_config,
                 step_hook=None, post_backward_hook=None):
    """Shared training loop for all discrete methods."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

    val0, ppl0 = evaluate(model, device)
    print(f"  [{method_name}] start: val={val0:.4f} ppl={ppl0:.1f}")

    wandb.init(project=WANDB_PROJECT, name=wandb_name,
               settings=wandb.Settings(init_timeout=180),
               config={**wandb_config, "method": method_name, "lr": LR,
                       "steps": PHASE2_STEPS, "density": DENSITY,
                       "d_model": D_MODEL, "n_layers": N_LAYERS})
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)

    loader = get_loader()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break

        if step_hook:
            step_hook(model, step, PHASE2_STEPS)

        lr_t = cosine_lr(step, WARMUP, PHASE2_STEPS, LR, LR_MIN)
        for g in optimizer.param_groups:
            g["lr"] = lr_t

        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        loss = out["loss"]
        loss.backward()

        if post_backward_hook:
            post_backward_hook(model)

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            elapsed = time.time() - t_start
            sps = step / elapsed
            wandb.log({"train/loss": avg, "train/lr": lr_t, "train/step_per_s": sps}, step=step)
            running_loss = 0.0
            if step % (LOG_EVERY * 10) == 0:
                print(f"    step {step}/{PHASE2_STEPS}  loss={avg:.4f}  lr={lr_t:.2e}  "
                      f"{sps:.1f} step/s")

        if step % EVAL_EVERY == 0:
            val, ppl = evaluate(model, device)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)
            print(f"    step {step}  val={val:.4f}  ppl={ppl:.1f}")

    val_f, ppl_f = evaluate(model, device)
    wandb.log({"val/loss": val_f, "val/ppl": ppl_f}, step=PHASE2_STEPS)
    elapsed = time.time() - t_start
    print(f"  [{method_name}] final: val={val_f:.4f} ppl={ppl_f:.1f}  ({elapsed:.0f}s)")
    wandb.finish()
    return val_f, ppl_f


# ---------------------------------------------------------------------------
# Experiment 0: STE+Adam bf16 baseline (continue Phase 1)
# ---------------------------------------------------------------------------

def run_exp0(device):
    print("\n=== Exp 0: STE+Adam bf16 baseline ===")
    model, cfg = load_phase1(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    val0, ppl0 = evaluate(model, device)
    print(f"  [ste_baseline] start: val={val0:.4f} ppl={ppl0:.1f}")

    wandb.init(project=WANDB_PROJECT, name=f"discrete_ste_bf16_d{DENSITY:.0%}",
               settings=wandb.Settings(init_timeout=180),
               config={"method": "ste_adam_bf16", "lr": LR, "steps": PHASE2_STEPS,
                       "density": DENSITY, "d_model": D_MODEL, "n_layers": N_LAYERS})
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)

    loader = get_loader()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE2_STEPS:
            break

        lr_t = cosine_lr(step, WARMUP, PHASE2_STEPS, LR, LR_MIN)
        for g in optimizer.param_groups:
            g["lr"] = lr_t

        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Quantize optimizer states to bf16 after each step
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state:
                    state["exp_avg"] = state["exp_avg"].to(torch.bfloat16)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.bfloat16)

        running_loss += out["loss"].item()

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            elapsed = time.time() - t_start
            sps = step / elapsed
            wandb.log({"train/loss": avg, "train/lr": lr_t, "train/step_per_s": sps}, step=step)
            running_loss = 0.0
            if step % (LOG_EVERY * 10) == 0:
                print(f"    step {step}/{PHASE2_STEPS}  loss={avg:.4f}  lr={lr_t:.2e}  "
                      f"{sps:.1f} step/s")

        if step % EVAL_EVERY == 0:
            val, ppl = evaluate(model, device)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)
            print(f"    step {step}  val={val:.4f}  ppl={ppl:.1f}")

    val_f, ppl_f = evaluate(model, device)
    wandb.log({"val/loss": val_f, "val/ppl": ppl_f}, step=PHASE2_STEPS)
    elapsed = time.time() - t_start
    print(f"  [ste_baseline] final: val={val_f:.4f} ppl={ppl_f:.1f}  ({elapsed:.0f}s)")
    wandb.finish()
    return val_f, ppl_f


# ---------------------------------------------------------------------------
# Experiment 1: Gumbel-Softmax (soft relaxation)
# ---------------------------------------------------------------------------

def run_exp1(device):
    from gumbel_ternary import GumbelTernaryLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 1: Gumbel-Softmax (soft) ===")
    model, cfg = load_phase1(device)
    pruned_data, _ = get_weight_data(model, device)
    gmodel = create_discrete_model(model, pruned_data, cfg, device, GumbelTernaryLinear)
    del model; torch.cuda.empty_cache()

    TAU_START, TAU_END = 2.0, 0.1

    def step_hook(m, step, total):
        tau = tau_anneal(step, total, TAU_START, TAU_END)
        for _, mod in m.discrete_modules:
            mod.tau = tau
            mod.hard = False

    return train_method(gmodel, device, "gumbel_soft", "discrete_gumbel_soft",
                        {"tau_start": TAU_START, "tau_end": TAU_END},
                        step_hook=step_hook)


# ---------------------------------------------------------------------------
# Experiment 2: Straight-Through Gumbel (hard forward + soft backward)
# ---------------------------------------------------------------------------

def run_exp2(device):
    from gumbel_ternary import GumbelTernaryLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 2: Straight-Through Gumbel (STG) ===")
    model, cfg = load_phase1(device)
    pruned_data, _ = get_weight_data(model, device)
    gmodel = create_discrete_model(model, pruned_data, cfg, device, GumbelTernaryLinear)
    del model; torch.cuda.empty_cache()

    TAU_START, TAU_END = 2.0, 0.5

    def step_hook(m, step, total):
        tau = tau_anneal(step, total, TAU_START, TAU_END)
        for _, mod in m.discrete_modules:
            mod.tau = tau
            mod.hard = True

    return train_method(gmodel, device, "stg", "discrete_stg",
                        {"tau_start": TAU_START, "tau_end": TAU_END},
                        step_hook=step_hook)


# ---------------------------------------------------------------------------
# Experiment 3: REINMAX (2nd-order gradient correction)
# ---------------------------------------------------------------------------

def run_exp3(device):
    from reinmax_ternary import ReinmaxTernaryLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 3: REINMAX ===")
    model, cfg = load_phase1(device)
    pruned_data, _ = get_weight_data(model, device)
    rmodel = create_discrete_model(model, pruned_data, cfg, device, ReinmaxTernaryLinear)
    del model; torch.cuda.empty_cache()

    return train_method(rmodel, device, "reinmax", "discrete_reinmax",
                        {"tau": 1.0})


# ---------------------------------------------------------------------------
# Experiment 4: Mirror Descent (probability-space Adam)
# ---------------------------------------------------------------------------

def run_exp4(device):
    from mirror_descent_ternary import MirrorDescentLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 4: Mirror Descent ===")
    model, cfg = load_phase1(device)
    pruned_data, _ = get_weight_data(model, device)
    mdmodel = create_discrete_model(model, pruned_data, cfg, device, MirrorDescentLinear)
    del model; torch.cuda.empty_cache()

    def post_backward_hook(m):
        for _, mod in m.discrete_modules:
            mod.transfer_gradients()

    return train_method(mdmodel, device, "mirror_descent", "discrete_mirror_descent",
                        {}, post_backward_hook=post_backward_hook)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    0: ("STE+Adam bf16 (baseline)", run_exp0),
    1: ("Gumbel-Softmax (soft)", run_exp1),
    2: ("Straight-Through Gumbel", run_exp2),
    3: ("REINMAX (2nd-order)", run_exp3),
    4: ("Mirror Descent", run_exp4),
}


def main():
    global DENSITY

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=None, help="Run single experiment (0-4)")
    parser.add_argument("--density", type=float, default=1.0, help="Weight density (1.0=full, 0.2=pruned)")
    args = parser.parse_args()

    DENSITY = args.density

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase 1 checkpoint: {PHASE1_CKPT}")
    print(f"Config: d={D_MODEL}, L={N_LAYERS}, B={B}, seq={SEQ}, "
          f"steps={PHASE2_STEPS}, density={DENSITY:.0%}")

    results = {}

    if args.exp is not None:
        name, fn = EXPERIMENTS[args.exp]
        print(f"\nRunning experiment {args.exp}: {name}")
        val, ppl = fn(device)
        results[args.exp] = (name, val, ppl)
    else:
        for exp_id in sorted(EXPERIMENTS.keys()):
            name, fn = EXPERIMENTS[exp_id]
            try:
                val, ppl = fn(device)
                results[exp_id] = (name, val, ppl)
            except Exception as e:
                print(f"  [FAILED] {name}: {e}")
                results[exp_id] = (name, float("nan"), float("nan"))
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"DISCRETE OPTIMIZATION SCOREBOARD (density={DENSITY:.0%})")
    print("=" * 70)
    print(f"{'#':<4} {'Method':<35} {'Val Loss':<10} {'Val PPL':<10}")
    print("-" * 70)
    for eid in sorted(results.keys()):
        name, val, ppl = results[eid]
        print(f"{eid:<4} {name:<35} {val:<10.4f} {ppl:<10.1f}")
    print("-" * 70)
    print(f"Phase 1 init: val=5.34 / PPL=209")
    print("=" * 70)


if __name__ == "__main__":
    main()
