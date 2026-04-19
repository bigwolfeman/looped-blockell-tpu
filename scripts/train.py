"""Three-phase training: Loop → Prune (with column reorder) → Route.

Usage:
    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --config configs/medium.yaml --wandb-run my_run
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
from looped_blockell.opt.tile_pruning import (
    create_tile_masks,
    apply_tile_masks,
    init_tile_scores,
    accumulate_tile_scores,
    normalize_tile_scores,
    prune_tiles,
    get_density,
    TileScores,
)


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
        n_coda=m.get("n_coda", 1), vocab_size=m.get("vocab_size", 50257),
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
    decay_fn = optax.cosine_decay_schedule(cfg.lr, cfg.total_steps - cfg.warmup_steps)
    schedule = optax.join_schedules([warmup_fn, decay_fn], [cfg.warmup_steps])
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(schedule, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95),
    )


def count_params(params) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def get_data_iterator(cfg: LoopedBlockELLConfig, key: jax.Array):
    """Synthetic data for testing. Replace with real data loading."""
    while True:
        key, subkey = jax.random.split(key)
        tokens = jax.random.randint(subkey, (cfg.batch_size, cfg.max_seq_len + 1), 0, cfg.vocab_size)
        yield tokens[:, :-1], tokens[:, 1:]


def train(cfg: LoopedBlockELLConfig, args):
    print(f"JAX devices: {jax.devices()}")
    print(f"Config: d={cfg.d_model}, heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"Layers: {cfg.n_prelude}p+{cfg.n_core}c+{cfg.n_coda}coda, "
          f"effective depth={cfg.effective_depth}")

    n_prune_rounds = max(0, (cfg.prune_end - cfg.prune_start) // cfg.prune_interval)
    final_density = (1 - cfg.prune_frac) ** n_prune_rounds
    print(f"Pruning: {n_prune_rounds} rounds, {cfg.prune_frac:.0%}/round → {final_density:.1%} target density")

    key = jax.random.PRNGKey(args.seed)
    key, init_key, data_key = jax.random.split(key, 3)

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
    n_params = count_params(params)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ─── Optimizer ───
    tx = create_optimizer(cfg)
    opt_state = tx.init(params)

    # ─── CMS tile masks + scores ───
    tile_masks = create_tile_masks(cfg.n_core, cfg.d_model, cfg.d_ff, cfg.tile_size)
    tile_scores = init_tile_scores(cfg.n_core, cfg.d_model, cfg.d_ff, cfg.tile_size)
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
                "tile_size": cfg.tile_size, "macro_tile_size": cfg.macro_tile_size,
            },
        )

    # ─── Jitted train step ───
    @partial(jax.jit, static_argnums=(4, 5))
    def train_step(params, opt_state, input_ids, labels, n_max, k_max, depths, step_key):
        dropout_key = jax.random.fold_in(step_key, 1)

        def loss_fn(p):
            out = model.apply(
                {"params": p},
                input_ids=input_ids, labels=labels,
                depths=depths, n_max=n_max, k_max=k_max,
                deterministic=False,
                rngs={"dropout": dropout_key},
            )
            return out["loss"], out

        (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grads

    # ─── Data ───
    data_iter = get_data_iterator(cfg, data_key)

    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")

    t0 = time.time()
    for step in range(cfg.total_steps):
        input_ids, labels = next(data_iter)

        # Apply tile masks to params (zero out pruned tiles before forward)
        masked_params = apply_tile_masks(params, tile_masks, cfg.tile_size)

        # Sample depth
        key, depth_key, step_key = jax.random.split(key, 3)
        if cfg.use_poisson:
            plan = sample_depth(
                depth_key, cfg.batch_size, cfg.mean_depth,
                min_depth=1, max_depth=cfg.max_depth, bptt_depth=cfg.bptt_depth
            )
        else:
            plan = sample_fixed(cfg.batch_size, cfg.mean_depth, cfg.bptt_depth)

        # Train step (on masked params)
        new_params, opt_state, loss, grads = train_step(
            masked_params, opt_state, input_ids, labels,
            plan.n_max, plan.k_max, plan.total, step_key
        )

        # ─── CMS scoring: accumulate gradient norms (between grad and next step) ───
        if cfg.prune_start <= step < cfg.prune_end:
            tile_scores = accumulate_tile_scores(tile_scores, grads, tile_masks, cfg.tile_size)

        # Update params (keep unmasked params for gradient flow on dead tiles)
        # Re-apply mask to ensure dead tiles stay zero
        params = apply_tile_masks(new_params, tile_masks, cfg.tile_size)

        loss_val = float(loss)
        ppl = math.exp(min(loss_val, 20.0))

        # Phase label
        if is_routing:
            phase = "ROUTE"
        elif prune_round > 0:
            phase = f"PRUNE R{prune_round}"
        else:
            phase = "DENSE"

        # ─── Score normalization every N steps ───
        if step > 0 and step % cfg.score_interval == 0:
            tile_scores = normalize_tile_scores(tile_scores)

        # ─── Pruning ───
        if (cfg.prune_start <= step < cfg.prune_end
                and step > 0
                and step % cfg.prune_interval == 0):
            prune_round += 1
            old_density = get_density(tile_masks)
            tile_masks, n_killed = prune_tiles(tile_masks, tile_scores, cfg.prune_frac)
            new_density = get_density(tile_masks)

            # Re-zero dead tiles in params and optimizer state
            params = apply_tile_masks(params, tile_masks, cfg.tile_size)

            # Reset scores after prune
            tile_scores = init_tile_scores(cfg.n_core, cfg.d_model, cfg.d_ff, cfg.tile_size)

            print(f"  PRUNE R{prune_round} @ step {step}: "
                  f"killed {n_killed} tiles, {old_density:.1%} → {new_density:.1%} density")

            if args.wandb_run:
                import wandb
                wandb.log({
                    "prune/round": prune_round,
                    "prune/density": new_density,
                    "prune/tiles_killed": n_killed,
                }, step=step)

        # ─── Compact + routing transition ───
        if step == cfg.prune_end and not is_routing:
            is_routing = True
            density = get_density(tile_masks)
            print(f"\n  {'='*60}")
            print(f"  COMPACT + ROUTE @ step {step}")
            print(f"  Final density: {density:.1%}")
            print(f"  Switching to Phase C: iteration-aware routing")
            print(f"  {'='*60}\n")
            # TODO: Physical compaction (masked-dense → Block-ELL)
            # TODO: Column reorder for macro-block density
            # TODO: Add router params to optimizer
            # TODO: Create iteration embeddings

        # ─── Logging ───
        if step % 10 == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            eta_h = (cfg.total_steps - step) / sps / 3600 if sps > 0 else 0
            density = get_density(tile_masks)

            print(
                f"step {step:6d} [{phase:>10s}] | "
                f"loss {loss_val:.4f} | ppl {ppl:8.2f} | "
                f"{sps:.1f} step/s | d={density:.2f} | "
                f"eta {eta_h:.1f}h"
            )

            if args.wandb_run:
                import wandb
                wandb.log({
                    "train/loss": loss_val, "train/ppl": ppl,
                    "train/steps_per_sec": sps,
                    "train/density": density,
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
    print(f"\n{'='*70}")
    print(f"Training complete: {cfg.total_steps} steps in {elapsed/3600:.1f}h")
    print(f"Final loss: {loss_val:.4f}, PPL: {ppl:.2f}, density: {get_density(tile_masks):.1%}")
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
