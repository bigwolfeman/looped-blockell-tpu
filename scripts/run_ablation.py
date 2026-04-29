"""Ablation runner with robust checkpointing and wandb resume.

Designed for quick architecture experiments on a single GPU/TPU.
Checkpoints every CKPT_INTERVAL steps. On restart, auto-resumes from
latest checkpoint and continues the same wandb run.

Usage:
    python scripts/run_ablation.py --config configs/ablation.yaml --name outer_ssm_detach
    # ctrl+c, restart later:
    python scripts/run_ablation.py --config configs/ablation.yaml --name outer_ssm_detach
    # ^ picks up where it left off
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from looped_blockell.config import LoopedBlockELLConfig
from looped_blockell.looping import LoopedTransformer, sample_depth, sample_fixed

CKPT_INTERVAL = 5000
EVAL_INTERVAL = 1000
LOG_INTERVAL = 50


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str, overlay: str | None = None) -> LoopedBlockELLConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overlay:
        with open(overlay) as f:
            over = yaml.safe_load(f)
        raw = _deep_merge(raw, over)
    m = raw.get("model", {})
    t = raw.get("training", {})
    return LoopedBlockELLConfig(
        d_model=m.get("d_model", 512),
        n_heads=m.get("n_heads", 8),
        d_ff=m.get("d_ff", 2048),
        n_layers=m.get("n_layers", 3),
        n_prelude=m.get("n_prelude", 1),
        n_core=m.get("n_core", 1),
        n_coda=m.get("n_coda", 1),
        vocab_size=m.get("vocab_size", 49152),
        max_seq_len=m.get("max_seq_len", 1024),
        mean_depth=m.get("mean_depth", 6),
        max_depth=m.get("max_depth", 12),
        use_poisson=m.get("use_poisson", True),
        init_decay=m.get("init_decay", 0.447),
        tile_size=m.get("tile_size", 16),
        macro_tile_size=m.get("macro_tile_size", 128),
        tie_weights=m.get("tie_weights", True),
        use_outer_ssm=m.get("use_outer_ssm", False),
        outer_state_detach=m.get("outer_state_detach", True),
        outer_init_decay=m.get("outer_init_decay", 0.447),
        embed_geometry=m.get("embed_geometry", "euclidean"),
        lorentz_dim_fraction=m.get("lorentz_dim_fraction", 0.5),
        use_loop_boundary_hc=m.get("use_loop_boundary_hc", False),
        hc_n_streams=m.get("hc_n_streams", 4),
        use_sparse_attention=m.get("use_sparse_attention", False),
        sparse_attn_type=m.get("sparse_attn_type", "dsa"),
        sparse_attn_top_k=m.get("sparse_attn_top_k", 256),
        sparse_attn_block_size=m.get("sparse_attn_block_size", 32),
        sparse_attn_n_indexer_heads=m.get("sparse_attn_n_indexer_heads", 4),
        csa_compress_ratio=m.get("csa_compress_ratio", 8),
        csa_compress_stride=m.get("csa_compress_stride", 4),
        csa_window_size=m.get("csa_window_size", 128),
        lr=float(t.get("lr", 6e-4)),
        weight_decay=t.get("weight_decay", 0.1),
        warmup_steps=t.get("warmup_steps", 500),
        total_steps=t.get("total_steps", 15000),
        batch_size=t.get("batch_size", 20),
        grad_clip=t.get("grad_clip", 1.0),
    )


# ─── Checkpointing ──────────────────────────────────────────────────────────

def _ckpt_dir(name: str) -> Path:
    d = Path("checkpoints/ablation") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(name: str, step: int, params, opt_state, key, outer_state=None):
    """Save training state. Keeps only latest checkpoint."""
    import flax.serialization as ser
    d = _ckpt_dir(name)
    ckpt = {
        "step": step,
        "params": ser.to_bytes(params),
        "opt_state": ser.to_bytes(opt_state),
        "key": np.array(key),
    }
    if outer_state is not None:
        ckpt["outer_state"] = np.array(outer_state)

    tmp = d / "ckpt_tmp.pkl"
    final = d / "ckpt.pkl"
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    tmp.rename(final)
    print(f"  💾 Checkpoint saved: step {step}")


def load_checkpoint(name: str, params_template, opt_state_template):
    """Load latest checkpoint. Returns None if no checkpoint exists."""
    import flax.serialization as ser
    ckpt_path = _ckpt_dir(name) / "ckpt.pkl"
    if not ckpt_path.exists():
        return None
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    params = ser.from_bytes(params_template, ckpt["params"])
    opt_state = ser.from_bytes(opt_state_template, ckpt["opt_state"])
    key = jnp.array(ckpt["key"])
    outer_state = jnp.array(ckpt["outer_state"]) if "outer_state" in ckpt else None
    print(f"  📂 Resumed from step {ckpt['step']}")
    return ckpt["step"], params, opt_state, key, outer_state


# ─── Data ────────────────────────────────────────────────────────────────────

class StreamingLoader:
    """Minimal streaming loader from Dolma with sequential consumption."""

    def __init__(self, batch_size: int, seq_len: int, buffer_tokens: int = 50_000_000):
        from transformers import AutoTokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.buffer_tokens = buffer_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starcoder2-7b", trust_remote_code=True
        )
        self.eos_id = self.tokenizer.eos_token_id
        self.buf = np.zeros(buffer_tokens, dtype=np.int32)
        self.buf_len = 0
        self.cursor = 0
        self.total_tokens = 0
        self.stream = None
        self._init_stream()
        self._fill()

    def _init_stream(self):
        from datasets import load_dataset
        self.stream = iter(load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        ))

    def _fill(self):
        tokens = []
        while len(tokens) < self.buffer_tokens:
            try:
                sample = next(self.stream)
            except StopIteration:
                self._init_stream()
                continue
            text = sample.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.eos_id)
            tokens.extend(ids)
        n = min(len(tokens), self.buffer_tokens)
        self.buf[:n] = np.array(tokens[:n], dtype=np.int32)
        self.buf_len = n
        self.cursor = 0
        print(f"  Buffer: {n:,} tokens (total: {self.total_tokens / 1e9:.2f}B)")

    def skip_tokens(self, n: int):
        """Skip n tokens to resume from checkpoint position."""
        while n > 0:
            skip = min(n, self.buf_len - self.cursor)
            self.cursor += skip
            self.total_tokens += skip
            n -= skip
            if self.cursor >= self.buf_len:
                self._fill()

    def get_batch(self):
        window = self.seq_len + 1
        seqs = []
        for _ in range(self.batch_size):
            if self.cursor + window > self.buf_len:
                self._fill()
            seqs.append(self.buf[self.cursor:self.cursor + window])
            self.cursor += self.seq_len
            self.total_tokens += self.seq_len
        seqs = np.stack(seqs)
        return jnp.asarray(seqs[:, :-1]), jnp.asarray(seqs[:, 1:])


# ─── Training ────────────────────────────────────────────────────────────────

def train(cfg: LoopedBlockELLConfig, args):
    import wandb

    print(f"Devices: {jax.devices()}")
    print(f"Config: d={cfg.d_model}, {cfg.n_prelude}p+{cfg.n_core}c+{cfg.n_coda}coda, "
          f"T={cfg.mean_depth}, outer_ssm={cfg.use_outer_ssm}, "
          f"embed={cfg.embed_geometry}, sparse_attn={cfg.use_sparse_attention}")

    key = jax.random.PRNGKey(args.seed)
    key, k_init, k_state = jax.random.split(key, 3)

    model = LoopedTransformer(cfg)
    dummy_ids = jnp.ones((cfg.batch_size, cfg.max_seq_len), dtype=jnp.int32)
    dummy_depths = jnp.full((cfg.batch_size,), cfg.mean_depth, dtype=jnp.int32)
    bptt = cfg.bptt_depth
    n_max_init = max(0, cfg.mean_depth - bptt)
    k_max_init = min(cfg.mean_depth, bptt)

    init_kwargs = dict(input_ids=dummy_ids, depths=dummy_depths,
                       n_max=n_max_init, k_max=k_max_init)
    if cfg.use_outer_ssm:
        init_kwargs["outer_state"] = jnp.zeros((cfg.batch_size, cfg.max_seq_len, cfg.d_model))

    variables = model.init({"params": k_init, "state": k_state}, **init_kwargs)
    params = variables["params"]
    topology = variables.get("topology", {})
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    warmup = optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps)
    decay = optax.cosine_decay_schedule(cfg.lr, decay_steps=1_000_000, alpha=0.1)
    schedule = optax.join_schedules([warmup, decay], [cfg.warmup_steps])
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(schedule, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95),
    )
    opt_state = tx.init(params)

    # ─── Resume ──────────────────────────────────────────────────────────
    start_step = 0
    outer_state = None
    if cfg.use_outer_ssm:
        outer_state = jnp.zeros((cfg.batch_size, cfg.max_seq_len, cfg.d_model))

    ckpt = load_checkpoint(args.name, params, opt_state)
    if ckpt is not None:
        start_step, params, opt_state, key, ckpt_outer = ckpt
        if ckpt_outer is not None:
            outer_state = ckpt_outer

    # ─── wandb (resume-aware) ────────────────────────────────────────────
    wandb_id_file = _ckpt_dir(args.name) / "wandb_id.txt"
    if wandb_id_file.exists():
        wandb_id = wandb_id_file.read_text().strip()
        wandb.init(project="looped-blockell-ablation", id=wandb_id, resume="must",
                   name=args.name)
        print(f"  wandb resumed: {wandb_id}")
    else:
        run = wandb.init(project="looped-blockell-ablation", name=args.name,
                         config={
                             "d_model": cfg.d_model, "n_core": cfg.n_core,
                             "mean_depth": cfg.mean_depth, "n_params": n_params,
                             "lr": cfg.lr, "total_steps": cfg.total_steps,
                             "batch_size": cfg.batch_size,
                             "use_outer_ssm": cfg.use_outer_ssm,
                             "outer_state_detach": cfg.outer_state_detach,
                             "embed_geometry": cfg.embed_geometry,
                             "use_sparse_attention": cfg.use_sparse_attention,
                         })
        wandb_id_file.write_text(run.id)
        print(f"  wandb new run: {run.id}")

    # ─── Data ────────────────────────────────────────────────────────────
    loader = StreamingLoader(cfg.batch_size, cfg.max_seq_len)
    if start_step > 0:
        skip = start_step * cfg.batch_size * cfg.max_seq_len
        print(f"  Skipping {skip / 1e6:.0f}M tokens to resume position...")
        loader.skip_tokens(skip)

    # ─── Eval buffer ─────────────────────────────────────────────────────
    eval_batches = []
    for _ in range(20):
        eval_batches.append(loader.get_batch())

    # ─── JIT ─────────────────────────────────────────────────────────────
    from functools import partial

    @partial(jax.jit, static_argnums=(5, 6))
    def train_step(params, opt_state, x, y, depths, n_max, k_max, step_key, outer_state_in):
        state_key = jax.random.fold_in(step_key, 0)

        def loss_fn(p):
            kwargs = dict(input_ids=x, depths=depths, n_max=n_max, k_max=k_max,
                          labels=y, deterministic=False)
            if outer_state_in is not None:
                kwargs["outer_state"] = outer_state_in
            out = model.apply({"params": p, "topology": topology}, rngs={"state": state_key}, **kwargs)
            return out["loss"], out

        (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state, out.get("outer_state_out")

    @jax.jit
    def eval_step(params, x, y, step_key, outer_state_in):
        depths = jnp.full((x.shape[0],), cfg.mean_depth, dtype=jnp.int32)
        n_max = max(0, cfg.mean_depth - cfg.bptt_depth)
        k_max = min(cfg.mean_depth, cfg.bptt_depth)
        kwargs = dict(input_ids=x, depths=depths, n_max=n_max, k_max=k_max,
                      labels=y, deterministic=True)
        if outer_state_in is not None:
            kwargs["outer_state"] = outer_state_in
        out = model.apply({"params": params, "topology": topology}, rngs={"state": step_key}, **kwargs)
        return out["loss"]

    # ─── Train loop ──────────────────────────────────────────────────────
    print(f"\nTraining: steps {start_step}→{cfg.total_steps}")
    print(f"Compiling first step (max_depth={cfg.max_depth}, may take 1-3 min)...")
    sys.stdout.flush()
    t0 = time.time()

    # Warmup: compile train_step by running one step, timing it
    _compile_start = time.time()

    for step in range(start_step, cfg.total_steps):
        x, y = loader.get_batch()
        key, depth_key, step_key = jax.random.split(key, 3)

        if cfg.use_poisson:
            plan = sample_depth(depth_key, cfg.batch_size, cfg.mean_depth,
                                max_depth=cfg.max_depth, bptt_depth=cfg.bptt_depth)
        else:
            plan = sample_fixed(cfg.batch_size, cfg.mean_depth, cfg.bptt_depth)

        # Pad n_max/k_max to multiples of 4 to reduce Poisson recompilations
        def _pad4(v):
            return ((v + 3) // 4) * 4
        padded_n = _pad4(plan.n_max)
        padded_k = _pad4(plan.k_max)

        loss, params, opt_state, new_outer = train_step(
            params, opt_state, x, y, plan.total, padded_n, padded_k, step_key,
            outer_state,
        )

        if cfg.use_outer_ssm and new_outer is not None:
            if cfg.outer_state_detach:
                outer_state = jax.lax.stop_gradient(new_outer)
            else:
                outer_state = new_outer

        # Report compilation time on first step
        if step == start_step:
            _compile_elapsed = time.time() - _compile_start
            print(f"  ✓ First step compiled in {_compile_elapsed:.1f}s")
            sys.stdout.flush()

        if step % LOG_INTERVAL == 0:
            loss_val = float(loss)
            ppl = math.exp(min(loss_val, 20.0))
            elapsed = time.time() - t0
            sps = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            eta_min = (cfg.total_steps - step) / max(sps, 0.01) / 60
            print(f"  step {step:6d} | loss {loss_val:.4f} | ppl {ppl:7.1f} | "
                  f"{sps:.1f} step/s | eta {eta_min:.0f}m")
            sys.stdout.flush()
            wandb.log({"train/loss": loss_val, "train/ppl": ppl,
                        "perf/steps_per_s": sps}, step=step)

        if step > 0 and step % EVAL_INTERVAL == 0:
            val_losses = []
            for xv, yv in eval_batches:
                key, ek = jax.random.split(key)
                val_losses.append(float(eval_step(params, xv, yv, ek, outer_state)))
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(val_loss, 20.0))
            print(f"  📊 EVAL step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

        if step > 0 and step % CKPT_INTERVAL == 0:
            save_checkpoint(args.name, step, params, opt_state, key, outer_state)

    # Final eval + checkpoint
    val_losses = []
    for xv, yv in eval_batches:
        key, ek = jax.random.split(key)
        val_losses.append(float(eval_step(params, xv, yv, ek, outer_state)))
    val_loss = sum(val_losses) / len(val_losses)
    val_ppl = math.exp(min(val_loss, 20.0))
    print(f"\n  Final: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=cfg.total_steps)
    wandb.summary.update({"final_val_ppl": val_ppl, "final_val_loss": val_loss,
                          "n_params": n_params})

    save_checkpoint(args.name, cfg.total_steps, params, opt_state, key, outer_state)
    wandb.finish()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Ablation runner with checkpointing")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--overlay", default=None, help="Override config (merged on top of base)")
    parser.add_argument("--name", required=True, help="Run name (also checkpoint dir name)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(load_config(args.config, args.overlay), args)


if __name__ == "__main__":
    main()
