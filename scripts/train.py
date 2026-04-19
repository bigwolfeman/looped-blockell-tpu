"""Three-phase training: Loop → Prune (with column reorder) → Route.

Block-ELL from the start (density=1.0 = all tiles active). CMS scores
gradient norms on Block-ELL values directly. Prune → compact → route.

Usage:
    python scripts/train.py --config configs/small.yaml --wandb-run my_run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import yaml
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looped_blockell.config import LoopedBlockELLConfig
from looped_blockell.looping import (
    LoopedTransformer,
    sample_depth,
    sample_fixed,
)
from looped_blockell.opt.cms import (
    CMSState,
    init_cms_state,
    accumulate_scores,
    score_step,
    prune_step,
    get_density,
)
from looped_blockell.opt.compaction import full_compaction, _get_nested, _set_nested
from looped_blockell.routing import init_lambda


def load_config(path: str) -> LoopedBlockELLConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    m = raw.get("model", {})
    t = raw.get("training", {})
    p = raw.get("pruning", {})
    r = raw.get("routing", {})
    return LoopedBlockELLConfig(
        d_model=m.get("d_model", 768), n_heads=m.get("n_heads", 12),
        d_ff=m.get("d_ff", 3072), n_layers=m.get("n_layers", 6),
        n_prelude=m.get("n_prelude", 1), n_core=m.get("n_core", 4),
        n_coda=m.get("n_coda", 1), vocab_size=m.get("vocab_size", 49152),
        max_seq_len=m.get("max_seq_len", 1024), mean_depth=m.get("mean_depth", 8),
        max_depth=m.get("max_depth", 32), use_poisson=m.get("use_poisson", True),
        init_decay=m.get("init_decay", 0.447), tile_size=m.get("tile_size", 16),
        macro_tile_size=m.get("macro_tile_size", 128),
        tie_weights=m.get("tie_weights", True), dropout=m.get("dropout", 0.0),
        lr=t.get("lr", 6e-4), weight_decay=t.get("weight_decay", 0.1),
        warmup_steps=t.get("warmup_steps", 2000), total_steps=t.get("total_steps", 50000),
        batch_size=t.get("batch_size", 8), grad_clip=t.get("grad_clip", 1.0),
        prune_start=p.get("prune_start", 0), prune_end=p.get("prune_end", 26000),
        prune_frac=p.get("prune_frac", 0.10), prune_interval=p.get("prune_interval", 2000),
        score_interval=p.get("score_interval", 10), topology_interval=p.get("topology_interval", 100),
        n_clusters=r.get("n_clusters", 16),
        route_target_sparsity=r.get("route_target_sparsity", 0.5),
        route_warmup=r.get("route_warmup", 5000), route_l1_weight=r.get("route_l1_weight", 0.01),
    )


def create_optimizer(cfg: LoopedBlockELLConfig):
    warmup_fn = optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps)
    # 1M-horizon cosine: never decays to 0 within our run
    decay_fn = optax.cosine_decay_schedule(cfg.lr, decay_steps=1_000_000, alpha=0.1)
    schedule = optax.join_schedules([warmup_fn, decay_fn], [cfg.warmup_steps])
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(schedule, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95),
    )


def count_params(params) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def get_data_iterator(cfg: LoopedBlockELLConfig):
    """OpenWebText data iterator with StarCoder2 tokenizer."""
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading StarCoder2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b", trust_remote_code=True)
    eos_id = tokenizer.eos_token_id

    print("Streaming OpenWebText...")
    ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

    TARGET_TOKENS = 200_000_000
    buf = []
    for sample in ds:
        ids = tokenizer.encode(sample["text"], add_special_tokens=False)
        ids.append(eos_id)
        buf.extend(ids)
        if len(buf) >= TARGET_TOKENS:
            break
    buf = np.array(buf[:TARGET_TOKENS], dtype=np.int32)
    print(f"Token buffer: {len(buf):,} tokens ({len(buf)/1e6:.0f}M)")

    rng = np.random.default_rng(42)
    seq_len = cfg.max_seq_len
    while True:
        starts = rng.integers(0, len(buf) - seq_len - 1, size=cfg.batch_size)
        seqs = np.stack([buf[s:s + seq_len + 1] for s in starts])
        x = jnp.asarray(seqs[:, :-1])
        y = jnp.asarray(seqs[:, 1:])
        yield x, y


# ---------------------------------------------------------------------------
# Block-ELL param path helpers
# ---------------------------------------------------------------------------

def find_block_ell_paths(params, prefix=""):
    """Find all Block-ELL 'values' param paths in the params pytree.

    Returns list of (dotted_path, layer_name, fc_name) tuples.
    E.g. ('core_0.mlp.fc1.values', 'core_0', 'fc1')
    """
    paths = []
    if isinstance(params, dict):
        for k, v in params.items():
            full = f"{prefix}.{k}" if prefix else k
            if k == "values" and isinstance(v, jnp.ndarray) and v.ndim == 4:
                # Found a Block-ELL values tensor [R, K, B, B]
                parts = full.split(".")
                # Extract layer and fc name: e.g. core_0.mlp.fc1.values
                layer_name = parts[0] if len(parts) >= 3 else prefix
                fc_name = parts[-2] if len(parts) >= 2 else "unknown"
                paths.append((full, layer_name, fc_name))
            else:
                paths.extend(find_block_ell_paths(v, full))
    return paths


get_nested = _get_nested
set_nested = _set_nested


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: LoopedBlockELLConfig, args):
    print(f"JAX devices: {jax.devices()}")
    print(f"Config: d={cfg.d_model}, heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"Layers: {cfg.n_prelude}p+{cfg.n_core}c+{cfg.n_coda}coda, "
          f"effective depth={cfg.effective_depth}")

    n_prune_rounds = max(0, (cfg.prune_end - cfg.prune_start) // cfg.prune_interval)
    final_density = (1 - cfg.prune_frac) ** n_prune_rounds
    print(f"Pruning: {n_prune_rounds} rounds, {cfg.prune_frac:.0%}/round → {final_density:.1%} target")

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    # ─── Model init ───
    model = LoopedTransformer(cfg)
    dummy_ids = jnp.ones((cfg.batch_size, cfg.max_seq_len), dtype=jnp.int32)
    dummy_depths = jnp.full((cfg.batch_size,), cfg.mean_depth, dtype=jnp.int32)
    bptt = cfg.bptt_depth
    n_max_init = max(0, cfg.mean_depth - bptt)
    k_max_init = min(cfg.mean_depth, bptt)

    variables = model.init(
        {"params": init_key},
        input_ids=dummy_ids, depths=dummy_depths,
        n_max=n_max_init, k_max=k_max_init, deterministic=True,
    )
    params = variables["params"]
    # Topology state (col_indices, alive_mask) — mutable, updated by CMS
    topology = variables.get("topology", {})

    n_params = count_params(params)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Find all Block-ELL value paths for CMS scoring
    bell_paths = find_block_ell_paths(params)
    print(f"Block-ELL layers: {len(bell_paths)}")
    for path, layer, fc in bell_paths:
        shape = get_nested(params, path).shape
        print(f"  {path}: {shape}")

    # ─── CMS states (one per Block-ELL values tensor) ───
    cms_states = {}
    for path, layer, fc in bell_paths:
        vals = get_nested(params, path)
        R, K = vals.shape[0], vals.shape[1]
        cms_states[path] = init_cms_state(R, K)

    # ─── Optimizer ───
    tx = create_optimizer(cfg)
    opt_state = tx.init(params)

    prune_round = 0
    is_routing = False

    # ─── Wandb ───
    if args.wandb_run:
        import wandb
        wandb.init(
            project="looped-blockell-scaling",
            name=args.wandb_run,
            config={
                "d_model": cfg.d_model, "n_heads": cfg.n_heads, "d_ff": cfg.d_ff,
                "n_prelude": cfg.n_prelude, "n_core": cfg.n_core, "n_coda": cfg.n_coda,
                "mean_depth": cfg.mean_depth, "effective_depth": cfg.effective_depth,
                "n_params": n_params, "lr": cfg.lr, "total_steps": cfg.total_steps,
                "batch_size": cfg.batch_size, "prune_frac": cfg.prune_frac,
                "prune_interval": cfg.prune_interval, "target_density": final_density,
            },
        )

    # ─── Jitted forward+backward (returns grads, does NOT apply optimizer) ───
    @partial(jax.jit, static_argnums=(3, 4))
    def grad_step(params, input_ids, labels, n_max, k_max, depths, step_key, topology):
        dropout_key = jax.random.fold_in(step_key, 1)

        def loss_fn(p):
            out = model.apply(
                {"params": p, "topology": topology},
                input_ids=input_ids, labels=labels,
                depths=depths, n_max=n_max, k_max=k_max,
                deterministic=False, use_iter_embed=is_routing,
                rngs={"dropout": dropout_key, "state": jax.random.fold_in(step_key, 2)},
                mutable=["topology"],
            )
            if isinstance(out, tuple):
                result, _ = out
            else:
                result = out
            return result["loss"], result

        (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, grads

    # ─── Jitted optimizer update (separate so CMS goes between) ───
    @jax.jit
    def apply_grads(params, opt_state, grads):
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # ─── Data ───
    data_iter = get_data_iterator(cfg)

    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")

    t0 = time.time()
    for step in range(cfg.total_steps):
        input_ids, labels = next(data_iter)

        # Sample depth
        key, depth_key, step_key = jax.random.split(key, 3)
        if cfg.use_poisson:
            plan = sample_depth(
                depth_key, cfg.batch_size, cfg.mean_depth,
                min_depth=1, max_depth=cfg.max_depth, bptt_depth=cfg.bptt_depth
            )
        else:
            plan = sample_fixed(cfg.batch_size, cfg.mean_depth, cfg.bptt_depth)

        # Forward + backward (grads only, optimizer NOT applied yet)
        loss, grads = grad_step(
            params, input_ids, labels,
            plan.n_max, plan.k_max, plan.total, step_key, topology
        )

        # ─── CMS scoring: BETWEEN grad and optimizer update (CRITICAL) ───
        if step <= cfg.prune_end:
            for path, layer, fc in bell_paths:
                grad_vals = get_nested(grads, path)
                cms_states[path] = accumulate_scores(cms_states[path], grad_vals)

        # ─── Apply optimizer ───
        params, opt_state = apply_grads(params, opt_state, grads)

        # ─── Re-zero dead tiles every step (prevents momentum resurrection) ───
        for path, layer, fc in bell_paths:
            alive_4d = cms_states[path].alive_mask[:, :, None, None]
            old_vals = get_nested(params, path)
            params = set_nested(params, path, old_vals * alive_4d)

        loss_val = float(loss)
        ppl = math.exp(min(loss_val, 20.0))

        # Phase label
        if is_routing:
            phase = "ROUTE"
        elif prune_round > 0:
            phase = f"PRUNE R{prune_round}"
        else:
            phase = "DENSE"

        # ─── Score normalization ───
        if step > 0 and step % cfg.score_interval == 0:
            for path in cms_states:
                cms_states[path] = score_step(cms_states[path])

        # ─── Pruning ───
        if (cfg.prune_start <= step < cfg.prune_end
                and step > 0
                and step % cfg.prune_interval == 0):
            prune_round += 1
            total_killed = 0

            for path, layer, fc in bell_paths:
                old_alive = int(cms_states[path].alive_mask.sum())

                cms_states[path], new_col_indices = prune_step(
                    cms_states[path],
                    get_nested(topology, f"{layer}.mlp.{fc}.col_indices"),
                    cfg.prune_frac,
                )

                topology = set_nested(
                    topology,
                    f"{layer}.mlp.{fc}.col_indices",
                    new_col_indices,
                )
                topology = set_nested(
                    topology,
                    f"{layer}.mlp.{fc}.alive_mask",
                    cms_states[path].alive_mask,
                )

                new_alive = int(cms_states[path].alive_mask.sum())
                total_killed += old_alive - new_alive

            # Reset scores after prune
            for path in cms_states:
                cms_states[path] = cms_states[path].replace(
                    gradient_scores=jnp.zeros_like(cms_states[path].gradient_scores),
                    total_score_steps=jnp.array(0, dtype=jnp.int32),
                )

            avg_density = sum(float(get_density(s)) for s in cms_states.values()) / len(cms_states)
            print(f"  PRUNE R{prune_round} @ step {step}: killed {total_killed} tiles, "
                  f"density={avg_density:.1%}")

            if args.wandb_run:
                import wandb
                wandb.log({"prune/round": prune_round, "prune/density": avg_density}, step=step)

        # ─── Compact + routing transition ───
        if step == cfg.prune_end and not is_routing:
            is_routing = True

            key, compact_key = jax.random.split(key)
            params, topology, opt_state, k_active_macros = full_compaction(
                params, topology, bell_paths, tx, cfg, compact_key
            )

            # Re-discover paths (shapes changed after compaction)
            bell_paths = find_block_ell_paths(params)

            # Re-init CMS states for new K
            cms_states = {}
            for path, layer, fc in bell_paths:
                vals = get_nested(params, path)
                R, K = vals.shape[0], vals.shape[1]
                cms_states[path] = init_cms_state(R, K)

            # Re-init routing lambdas
            route_lambdas = {}
            for path, layer, fc in bell_paths:
                route_lambdas[path] = init_lambda(cfg.n_clusters)

            if args.wandb_run:
                import wandb
                wandb.log({
                    "compact/step": step,
                    **{f"compact/k_active_{k}": v for k, v in k_active_macros.items()},
                }, step=step)

        # ─── Logging ───
        if step % 10 == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            eta_h = (cfg.total_steps - step) / sps / 3600 if sps > 0 else 0
            avg_density = sum(float(get_density(s)) for s in cms_states.values()) / max(len(cms_states), 1)

            print(
                f"step {step:6d} [{phase:>10s}] | "
                f"loss {loss_val:.4f} | ppl {ppl:8.2f} | "
                f"{sps:.1f} step/s | d={avg_density:.2f} | "
                f"eta {eta_h:.1f}h"
            )

            if args.wandb_run:
                import wandb
                wandb.log({
                    "train/loss": loss_val, "train/ppl": ppl,
                    "train/steps_per_sec": sps, "train/density": avg_density,
                    "depth/n_max": plan.n_max, "depth/k_max": plan.k_max,
                }, step=step)

        # ─── Checkpoint ───
        if args.checkpoint_dir and step > 0 and step % args.save_interval == 0:
            ckpt_dir = Path(args.checkpoint_dir) / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            import orbax.checkpoint as ocp
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(ckpt_dir / "params", params)
            print(f"  Checkpoint saved: {ckpt_dir}")

    elapsed = time.time() - t0
    avg_density = sum(float(get_density(s)) for s in cms_states.values()) / max(len(cms_states), 1)
    print(f"\n{'='*70}")
    print(f"Training complete: {cfg.total_steps} steps in {elapsed/3600:.1f}h")
    print(f"Final loss: {loss_val:.4f}, PPL: {ppl:.2f}, density: {avg_density:.1%}")
    print(f"{'='*70}")

    if args.wandb_run:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Looped Block-ELL Transformer on TPU")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--wandb-run", default=None, help="Wandb run name")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory")
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(load_config(args.config), args)


if __name__ == "__main__":
    main()
