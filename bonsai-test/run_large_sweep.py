"""Large-scale discrete ternary sweep — d=768, L=6, 118M params.

Trains Phase 1 from scratch (STE+Adam, 5k steps), then runs:
  0. STE+Adam bf16 baseline (20k steps)
  1. Gumbel-Softmax (soft)
  2. Straight-Through Gumbel (STG)
  3. REINMAX (2nd-order)
  4. Mirror Descent (prob-space Adam)

All at full density (no pruning). Fair comparison of gradient estimators.

Usage:
  python run_large_sweep.py               # train Phase 1 + all experiments
  python run_large_sweep.py --exp 0       # just STE baseline (assumes Phase 1 exists)
  python run_large_sweep.py --phase1-only # just train Phase 1
  python run_large_sweep.py --skip-phase1 # skip Phase 1, run experiments
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import wandb

from model import TernaryTransformer, TernaryConfig
from bitlinear import BitLinear, BitEmbedding, ternary_quantize

torch.set_float32_matmul_precision("high")
os.environ["PYTHONUNBUFFERED"] = "1"

WANDB_PROJECT = "bonsai-ternary-test"
B = 32
SEQ = 256
VOCAB = 49152

D_MODEL = 768
N_HEADS = 12
D_FF = 2048
N_LAYERS = 6

PHASE1_STEPS = 5000
PHASE2_STEPS = 20000
LR = 3e-4
LR_MIN = 3e-5
WARMUP = 500

CKPT_PATH = "checkpoints/discrete_phase1_d768L6.pt"
LOG_EVERY = 50
EVAL_EVERY = 1000


def get_loader():
    from run_ab_eggroll import get_data_loader
    return get_data_loader(batch_size=B, seq_len=SEQ)


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def tau_anneal(step, total, tau_start, tau_end):
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


def full_density_data(model, device):
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


def load_or_train_phase1(device):
    cfg = TernaryConfig(
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, vocab_size=VOCAB, max_seq_len=1024,
    )

    if os.path.exists(CKPT_PATH):
        print(f"Loading Phase 1 checkpoint: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model = TernaryTransformer(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt["config"]

    print(f"\n=== Phase 1: STE+Adam training from scratch ===")
    print(f"Config: d={D_MODEL}, L={N_LAYERS}, d_ff={D_FF}, heads={N_HEADS}, B={B}")
    model = TernaryTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    wandb.init(project=WANDB_PROJECT, name="large_phase1_d768L6",
               settings=wandb.Settings(init_timeout=180),
               config={"phase": 1, "d_model": D_MODEL, "n_layers": N_LAYERS,
                       "batch_size": B, "steps": PHASE1_STEPS, "lr": LR})

    val0, ppl0 = evaluate(model, device)
    wandb.log({"val/loss": val0, "val/ppl": ppl0}, step=0)
    print(f"  init: val={val0:.4f} ppl={ppl0:.1f}")

    loader = get_loader()
    running_loss = 0.0
    t_start = time.time()

    for step, (x, y) in enumerate(loader, 1):
        if step > PHASE1_STEPS:
            break

        lr_t = cosine_lr(step, WARMUP, PHASE1_STEPS, LR, LR_MIN)
        for g in optimizer.param_groups:
            g["lr"] = lr_t

        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += out["loss"].item()

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            elapsed = time.time() - t_start
            sps = step / elapsed
            wandb.log({"train/loss": avg, "train/lr": lr_t, "train/step_per_s": sps}, step=step)
            running_loss = 0.0
            if step % (LOG_EVERY * 10) == 0:
                print(f"    step {step}/{PHASE1_STEPS}  loss={avg:.4f}  lr={lr_t:.2e}  "
                      f"{sps:.1f} step/s")

        if step % EVAL_EVERY == 0:
            val, ppl = evaluate(model, device)
            wandb.log({"val/loss": val, "val/ppl": ppl}, step=step)
            print(f"    step {step}  val={val:.4f}  ppl={ppl:.1f}")

    val_f, ppl_f = evaluate(model, device)
    elapsed = time.time() - t_start
    print(f"  Phase 1 done: val={val_f:.4f} ppl={ppl_f:.1f}  ({elapsed:.0f}s)")
    wandb.log({"val/loss": val_f, "val/ppl": ppl_f}, step=PHASE1_STEPS)
    wandb.finish()

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                "step": PHASE1_STEPS}, CKPT_PATH)
    print(f"  Saved: {CKPT_PATH}")

    return model, cfg


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def train_method(model, device, method_name, wandb_name, wandb_config,
                 step_hook=None, post_backward_hook=None):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

    val0, ppl0 = evaluate(model, device)
    print(f"  [{method_name}] start: val={val0:.4f} ppl={ppl0:.1f}")

    wandb.init(project=WANDB_PROJECT, name=wandb_name,
               settings=wandb.Settings(init_timeout=180),
               config={**wandb_config, "method": method_name, "lr": LR,
                       "steps": PHASE2_STEPS, "d_model": D_MODEL, "n_layers": N_LAYERS,
                       "batch_size": B, "total_params": "118M"})
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
# Experiments
# ---------------------------------------------------------------------------

def fresh_model(device):
    """Load a fresh copy of the Phase 1 model."""
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TernaryTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def run_exp0(device):
    """STE+Adam bf16 baseline — continue Phase 1 with bf16 optimizer states."""
    print("\n=== Exp 0: STE+Adam bf16 baseline ===")
    model, cfg = fresh_model(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    val0, ppl0 = evaluate(model, device)
    print(f"  [ste_bf16] start: val={val0:.4f} ppl={ppl0:.1f}")

    wandb.init(project=WANDB_PROJECT, name="large_ste_bf16",
               settings=wandb.Settings(init_timeout=180),
               config={"method": "ste_adam_bf16", "lr": LR, "steps": PHASE2_STEPS,
                       "d_model": D_MODEL, "n_layers": N_LAYERS, "batch_size": B})
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
    print(f"  [ste_bf16] final: val={val_f:.4f} ppl={ppl_f:.1f}  ({elapsed:.0f}s)")
    wandb.finish()
    return val_f, ppl_f


def run_exp1(device):
    from gumbel_ternary import GumbelTernaryLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 1: Gumbel-Softmax (soft) ===")
    model, cfg = fresh_model(device)
    wd = full_density_data(model, device)
    gmodel = create_discrete_model(model, wd, cfg, device, GumbelTernaryLinear)
    del model; torch.cuda.empty_cache()

    TAU_START, TAU_END = 2.0, 0.1

    def step_hook(m, step, total):
        tau = tau_anneal(step, total, TAU_START, TAU_END)
        for _, mod in m.discrete_modules:
            mod.tau = tau
            mod.hard = False

    return train_method(gmodel, device, "gumbel_soft", "large_gumbel_soft",
                        {"tau_start": TAU_START, "tau_end": TAU_END},
                        step_hook=step_hook)


def run_exp2(device):
    from gumbel_ternary import GumbelTernaryLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 2: Straight-Through Gumbel (STG) ===")
    model, cfg = fresh_model(device)
    wd = full_density_data(model, device)
    gmodel = create_discrete_model(model, wd, cfg, device, GumbelTernaryLinear)
    del model; torch.cuda.empty_cache()

    TAU_START, TAU_END = 2.0, 0.5

    def step_hook(m, step, total):
        tau = tau_anneal(step, total, TAU_START, TAU_END)
        for _, mod in m.discrete_modules:
            mod.tau = tau
            mod.hard = True

    return train_method(gmodel, device, "stg", "large_stg",
                        {"tau_start": TAU_START, "tau_end": TAU_END},
                        step_hook=step_hook)


def run_exp3(device):
    from reinmax_ternary import ReinmaxTernaryLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 3: REINMAX ===")
    model, cfg = fresh_model(device)
    wd = full_density_data(model, device)
    rmodel = create_discrete_model(model, wd, cfg, device, ReinmaxTernaryLinear)
    del model; torch.cuda.empty_cache()

    return train_method(rmodel, device, "reinmax", "large_reinmax", {"tau": 1.0})


def run_exp4(device):
    from mirror_descent_ternary import MirrorDescentLinear
    from discrete_ternary_base import create_discrete_model

    print("\n=== Exp 4: Mirror Descent ===")
    model, cfg = fresh_model(device)
    wd = full_density_data(model, device)
    mdmodel = create_discrete_model(model, wd, cfg, device, MirrorDescentLinear)
    del model; torch.cuda.empty_cache()

    def post_backward_hook(m):
        for _, mod in m.discrete_modules:
            mod.transfer_gradients()

    return train_method(mdmodel, device, "mirror_descent", "large_mirror_descent",
                        {}, post_backward_hook=post_backward_hook)


# ---------------------------------------------------------------------------
# Experiment 5: bf16 full (no ternary) — ceiling
# ---------------------------------------------------------------------------

def run_exp5(device):
    """bf16 baseline — skip ternary quantization entirely."""
    print("\n=== Exp 5: bf16 full (no ternary constraint) ===")
    model, cfg = fresh_model(device)
    model.train()

    # Monkey-patch BitLinear to skip STE quantization
    from bitlinear import BitLinear, BitEmbedding
    def bf16_forward(self, x):
        return F.linear(x, self.weight, self.bias)
    def bf16_embed_forward(self, input_ids):
        return F.embedding(input_ids, self.weight)
    for mod in model.modules():
        if isinstance(mod, BitLinear):
            mod.forward = bf16_forward.__get__(mod, BitLinear)
        elif isinstance(mod, BitEmbedding):
            mod.forward = bf16_embed_forward.__get__(mod, BitEmbedding)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    val0, ppl0 = evaluate(model, device)
    print(f"  [bf16_full] start: val={val0:.4f} ppl={ppl0:.1f}")

    wandb.init(project=WANDB_PROJECT, name="large_bf16_full",
               settings=wandb.Settings(init_timeout=180),
               config={"method": "bf16_full", "lr": LR, "steps": PHASE2_STEPS,
                       "d_model": D_MODEL, "n_layers": N_LAYERS, "batch_size": B})
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
    print(f"  [bf16_full] final: val={val_f:.4f} ppl={ppl_f:.1f}  ({elapsed:.0f}s)")
    wandb.finish()
    return val_f, ppl_f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    0: ("STE+Adam (ternary baseline)", run_exp0),
    1: ("Gumbel-Softmax (soft)", run_exp1),
    2: ("Straight-Through Gumbel", run_exp2),
    3: ("REINMAX (2nd-order)", run_exp3),
    4: ("Mirror Descent", run_exp4),
    5: ("bf16 full (no ternary)", run_exp5),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=None, help="Run single experiment (0-4)")
    parser.add_argument("--phase1-only", action="store_true", help="Only train Phase 1")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1 training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: d={D_MODEL}, L={N_LAYERS}, d_ff={D_FF}, heads={N_HEADS}, "
          f"B={B}, seq={SEQ}, ~118M params")

    if not args.skip_phase1:
        load_or_train_phase1(device)
        torch.cuda.empty_cache()

    if args.phase1_only:
        return

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
                import traceback; traceback.print_exc()
                results[exp_id] = (name, float("nan"), float("nan"))
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"LARGE-SCALE DISCRETE SCOREBOARD (d={D_MODEL}, L={N_LAYERS}, 118M params)")
    print("=" * 70)
    print(f"{'#':<4} {'Method':<35} {'Val Loss':<10} {'Val PPL':<10}")
    print("-" * 70)
    for eid in sorted(results.keys()):
        name, val, ppl = results[eid]
        print(f"{eid:<4} {name:<35} {val:<10.4f} {ppl:<10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
