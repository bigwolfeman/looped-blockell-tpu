"""PyTorch ablation runner — fast iteration on GPU.

Mirrors the JAX run_ablation.py but runs on PyTorch for 10-20× faster
step/s on the 5090. Checkpoints are interop-compatible: can be converted
to JAX format for TPU continuation via interop/convert_checkpoint.py.

Usage:
    python scripts/run_ablation_pt.py --config configs/ablation.yaml --name baseline
    python scripts/run_ablation_pt.py --config configs/ablation.yaml --overlay configs/ablations/outer_ssm_detach.yaml --name outer_ssm_detach
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch._functorch.config
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interop.pt_model import (
    InteropConfig,
    LoopedTransformerPT,
    PrunableLinear,
    DepthPlan,
    sample_depth,
    sample_fixed,
)

CKPT_INTERVAL = 2000
EVAL_INTERVAL = 500
LOG_INTERVAL = 20
SCORE_INTERVAL = 10


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str, overlay: str | None = None) -> InteropConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overlay:
        with open(overlay) as f:
            over = yaml.safe_load(f)
        raw = _deep_merge(raw, over)
    m = raw.get("model", {})
    t = raw.get("training", {})
    cfg = InteropConfig(
        d_model=m.get("d_model", 512),
        n_heads=m.get("n_heads", 8),
        d_ff=m.get("d_ff", 2048),
        n_prelude=m.get("n_prelude", 4),
        n_core=m.get("n_core", 8),
        n_coda=m.get("n_coda", 4),
        vocab_size=m.get("vocab_size", 49152),
        max_seq_len=m.get("max_seq_len", 1024),
        mean_depth=m.get("mean_depth", 6),
        max_depth=m.get("max_depth", 8),
        use_poisson=m.get("use_poisson", True),
        init_decay=m.get("init_decay", 0.447),
        use_outer_ssm=m.get("use_outer_ssm", False),
        outer_state_detach=m.get("outer_state_detach", True),
        outer_init_decay=m.get("outer_init_decay", 0.447),
        embed_geometry=m.get("embed_geometry", "euclidean"),
        lorentz_dim_fraction=m.get("lorentz_dim_fraction", 0.5),
        use_xsa=m.get("use_xsa", False),
        use_loop_boundary_hc=m.get("use_loop_boundary_hc", False),
        use_attn_res=m.get("use_attn_res", False),
        attn_res_window=m.get("attn_res_window", 3),
        hc_type=m.get("hc_type", "diagonal"),
        hc_n_streams=m.get("hc_n_streams", 4),
        use_sparse_attention=m.get("use_sparse_attention", False),
        sparse_attn_type=m.get("sparse_attn_type", "csa"),
        sparse_attn_top_k=m.get("sparse_attn_top_k", 256),
        sparse_attn_block_size=m.get("sparse_attn_block_size", 32),
        sparse_attn_n_indexer_heads=m.get("sparse_attn_n_indexer_heads", 4),
        csa_compress_ratio=m.get("csa_compress_ratio", 8),
        csa_compress_stride=m.get("csa_compress_stride", 4),
        csa_window_size=m.get("csa_window_size", 128),
        use_swiglu=m.get("use_swiglu", False),
        use_qk_norm=m.get("use_qk_norm", False),
        n_kv_heads=m.get("n_kv_heads", None),
        use_cla=m.get("use_cla", False),
        use_neural_memory=m.get("use_neural_memory", False),
        n_memory_layers=m.get("n_memory_layers", 6),
        d_memory=m.get("d_memory", 1024),
        memory_mode=m.get("memory_mode", "logit_bias"),
        memory_theta_lr=float(m.get("memory_theta_lr", 0.001)),
        memory_alpha_min=float(m.get("memory_alpha_min", 0.001)),
        memory_alpha_max=float(m.get("memory_alpha_max", 0.03)),
        memory_surprise_scale=float(m.get("memory_surprise_scale", 3.0)),
        memory_eta_fixed=float(m.get("memory_eta_fixed", 0.95)),
        use_sigreg=m.get("use_sigreg", True),
        sigreg_lambda=float(m.get("sigreg_lambda", 0.02)),
        use_differentiable_memory=m.get("use_differentiable_memory", False),
        memory_append_tokens=m.get("memory_append_tokens", 8),
        memory_inner_steps=m.get("memory_inner_steps", 1),
        memory_warmup_steps=m.get("memory_warmup_steps", 1000),
        memory_ramp_steps=m.get("memory_ramp_steps", 4000),
        memory_update_interval=m.get("memory_update_interval", 1),
        use_mtp=m.get("use_mtp", False),
        mtp_n_heads=m.get("mtp_n_heads", 3),
        mtp_lambda=float(m.get("mtp_lambda", 0.1)),
        enable_pruning=m.get("enable_pruning", False),
        tile_size=m.get("tile_size", 16),
        enable_routing=m.get("enable_routing", False),
        n_clusters=m.get("n_clusters", 16),
        lr=float(t.get("lr", 6e-4)),
        weight_decay=t.get("weight_decay", 0.1),
        warmup_steps=t.get("warmup_steps", 500),
        total_steps=t.get("total_steps", 15000),
        batch_size=t.get("batch_size", 20),
        grad_clip=t.get("grad_clip", 1.0),
    )
    # Pipeline schedule — stored as extra attrs (not in dataclass)
    cfg.prune_start = t.get("prune_start", 10000)
    cfg.prune_end = t.get("prune_end", 50000)
    cfg.prune_interval = t.get("prune_interval", 3000)
    cfg.prune_frac = t.get("prune_frac", 0.10)
    cfg.route_start = t.get("route_start", 52000)
    cfg.route_target_sparsity = float(t.get("route_target_sparsity", 0.5))
    cfg.route_warmup = t.get("route_warmup", 5000)
    cfg.route_l1_weight = float(t.get("route_l1_weight", 0.01))
    return cfg


# ─── Checkpointing ────────────────────────────────────────────────────────────

def _ckpt_dir(name: str) -> Path:
    d = Path("checkpoints/ablation") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(name: str, step: int, model, optimizer, outer_state=None,
                    prune_round: int = 0, current_density: float = 1.0):
    d = _ckpt_dir(name)
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "prune_round": prune_round,
        "current_density": current_density,
    }
    if outer_state is not None:
        ckpt["outer_state"] = outer_state.detach().cpu()
    tmp = d / "ckpt_tmp.pt"
    final = d / "ckpt.pt"
    torch.save(ckpt, tmp)
    tmp.rename(final)
    print(f"  Checkpoint saved: step {step}")


def load_checkpoint(name: str, model, optimizer, device):
    ckpt_path = _ckpt_dir(name) / "ckpt.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    outer = ckpt.get("outer_state")
    if outer is not None:
        outer = outer.to(device)
    prune_round = ckpt.get("prune_round", -1)
    current_density = ckpt.get("current_density", -1.0)
    if prune_round < 0:
        # Legacy checkpoint: infer from step and tile masks
        step = ckpt["step"]
        sd = ckpt["model_state_dict"]
        masks = [v for k, v in sd.items() if "tile_mask" in k and "core" in k]
        if masks:
            total_alive = sum(m.sum().item() for m in masks)
            total_tiles = sum(m.numel() for m in masks)
            current_density = total_alive / total_tiles
        else:
            current_density = 1.0
        prune_round = max(0, round(math.log(current_density) / math.log(0.9))) if current_density < 1.0 else 0
    print(f"  Resumed from step {ckpt['step']} "
          f"(prune_round={prune_round}, density={current_density:.1%})")
    return ckpt["step"], outer, prune_round, current_density


# ─── Data ──────────────────────────────────────────────────────────────────────

class StreamingLoader:
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
        while n > 0:
            skip = min(n, self.buf_len - self.cursor)
            self.cursor += skip
            self.total_tokens += skip
            n -= skip
            if self.cursor >= self.buf_len:
                self._fill()

    def get_batch(self, device: torch.device):
        window = self.seq_len + 1
        seqs = []
        for _ in range(self.batch_size):
            if self.cursor + window > self.buf_len:
                self._fill()
            seqs.append(self.buf[self.cursor:self.cursor + window])
            self.cursor += self.seq_len
            self.total_tokens += self.seq_len
        seqs = np.stack(seqs)
        x = torch.from_numpy(seqs[:, :-1].copy()).to(device, dtype=torch.long)
        y = torch.from_numpy(seqs[:, 1:].copy()).to(device, dtype=torch.long)
        return x, y


# ─── LR Schedule ───────────────────────────────────────────────────────────────

def _lr_schedule(step: int, cfg: InteropConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    decay_steps = 1_000_000
    progress = min((step - cfg.warmup_steps) / decay_steps, 1.0)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ─── Training ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, eval_batches, cfg, device, outer_state):
    model.eval()
    torch.cuda.empty_cache()
    os = outer_state.detach() if outer_state is not None else None
    losses = []
    for x, y in eval_batches:
        depths = torch.full((x.shape[0],), cfg.mean_depth, dtype=torch.int32, device=device)
        n = max(0, cfg.mean_depth - cfg.bptt_depth)
        k = min(cfg.mean_depth, cfg.bptt_depth)
        out = model(x, depths, n, k, labels=y, deterministic=True, outer_state=os)
        losses.append(out["loss"].item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: InteropConfig, args):
    import wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: d={cfg.d_model}, {cfg.n_prelude}p+{cfg.n_core}c+{cfg.n_coda}coda, "
          f"T={cfg.mean_depth}, outer_ssm={cfg.use_outer_ssm}, "
          f"embed={cfg.embed_geometry}")

    model = LoopedTransformerPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    if cfg.use_neural_memory:
        mem_params = sum(p.numel() for p in model.neural_memory.memory_mlp.parameters())
        print(f"  Memory MLP: {mem_params:,} ({mem_params / 1e6:.1f}M) — updated via inner loop")

    # Optimizer: exclude memory MLP (updated exclusively by inner loop)
    if cfg.use_neural_memory:
        mem_mlp_ids = {id(p) for p in model.neural_memory.memory_mlp.parameters()}
        outer_params = [p for p in model.parameters() if id(p) not in mem_mlp_ids and p.requires_grad]
        optimizer = torch.optim.AdamW(
            outer_params, lr=cfg.lr, betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay, eps=1e-8,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay, eps=1e-8,
        )

    # Memory update moved out of forward() → no graph break → donated_buffer OK
    if cfg.use_neural_memory:
        torch._functorch.config.donated_buffer = False  # retrieve still uses no_grad internals

    # torch.compile — use "default" mode (NOT "reduce-overhead" which uses
    # CUDA graphs that reserve ~15GB and cause OOM during eval)
    compiled_model = model
    try:
        compiled_model = torch.compile(model)
        print("  torch.compile enabled (default mode)")
    except Exception as e:
        print(f"  torch.compile failed ({e}), using eager mode")
        compiled_model = model

    # Resume
    start_step = 0
    outer_state = None
    prune_round = 0
    current_density = 1.0
    if cfg.use_outer_ssm:
        outer_state = torch.zeros(cfg.batch_size, cfg.max_seq_len, cfg.d_model,
                                  device=device, dtype=torch.bfloat16)

    ckpt = load_checkpoint(args.name, model, optimizer, device)
    if ckpt is not None:
        start_step, ckpt_outer, prune_round, current_density = ckpt
        if ckpt_outer is not None:
            outer_state = ckpt_outer

    # wandb
    wandb_id_file = _ckpt_dir(args.name) / "wandb_id.txt"
    if wandb_id_file.exists():
        wandb_id = wandb_id_file.read_text().strip()
        wandb.init(project="looped-blockell-ablation", id=wandb_id, resume="must",
                   name=args.name)
        print(f"  wandb resumed: {wandb_id}")
    else:
        run = wandb.init(
            project="looped-blockell-ablation", name=args.name,
            config={
                "d_model": cfg.d_model, "n_prelude": cfg.n_prelude,
                "n_core": cfg.n_core, "n_coda": cfg.n_coda,
                "mean_depth": cfg.mean_depth, "n_params": n_params,
                "lr": cfg.lr, "total_steps": cfg.total_steps,
                "batch_size": cfg.batch_size,
                "use_outer_ssm": cfg.use_outer_ssm,
                "outer_state_detach": cfg.outer_state_detach,
                "embed_geometry": cfg.embed_geometry,
                "use_loop_boundary_hc": cfg.use_loop_boundary_hc,
                "use_neural_memory": cfg.use_neural_memory,
                "memory_mode": cfg.memory_mode if cfg.use_neural_memory else None,
                "n_memory_layers": cfg.n_memory_layers if cfg.use_neural_memory else None,
                "d_memory": cfg.d_memory if cfg.use_neural_memory else None,
                "use_sigreg": cfg.use_sigreg if cfg.use_neural_memory else None,
                "use_xsa": cfg.use_xsa,
                "use_attn_res": cfg.use_attn_res,
                "use_mtp": cfg.use_mtp,
                "mtp_n_heads": cfg.mtp_n_heads if cfg.use_mtp else None,
                "mtp_lambda": cfg.mtp_lambda if cfg.use_mtp else None,
                "enable_pruning": cfg.enable_pruning,
                "tile_size": cfg.tile_size if cfg.enable_pruning else None,
                "prune_start": cfg.prune_start if cfg.enable_pruning else None,
                "prune_end": cfg.prune_end if cfg.enable_pruning else None,
                "prune_interval": cfg.prune_interval if cfg.enable_pruning else None,
                "prune_frac": cfg.prune_frac if cfg.enable_pruning else None,
                "enable_routing": cfg.enable_routing,
                "n_clusters": cfg.n_clusters if cfg.enable_routing else None,
                "route_start": cfg.route_start if cfg.enable_routing else None,
                "framework": "pytorch",
            },
        )
        wandb_id_file.write_text(run.id)
        print(f"  wandb new run: {run.id}")

    # Data
    loader = StreamingLoader(cfg.batch_size, cfg.max_seq_len)
    if start_step > 0:
        skip = start_step * cfg.batch_size * cfg.max_seq_len
        print(f"  Skipping {skip / 1e6:.0f}M tokens to resume position...")
        loader.skip_tokens(skip)

    # Eval buffer (20 batches)
    eval_batches = [loader.get_batch(device) for _ in range(20)]

    # AMP scaler for bf16
    use_amp = device.type == "cuda"

    # ─── Pruning / routing state ────────────────────────────────────────────
    # prune_round and current_density are set by load_checkpoint (or defaults above)
    prunable_mods = model.get_prunable_modules() if cfg.enable_pruning else []
    routing_active = False
    routed_blocks = []

    if prunable_mods:
        total_tiles = sum(m.R * m.C for _, _, m in prunable_mods)
        print(f"  Prunable modules: {len(prunable_mods)} ({total_tiles:,} tiles, "
              f"tile_size={cfg.tile_size})")
        max_rounds = (cfg.prune_end - cfg.prune_start) // max(cfg.prune_interval, 1)
        target_density = (1 - cfg.prune_frac) ** max_rounds
        print(f"  Prune schedule: steps {cfg.prune_start}→{cfg.prune_end}, "
              f"{cfg.prune_frac:.0%}/round every {cfg.prune_interval} steps, "
              f"{max_rounds} rounds → ~{target_density:.1%} target density")
        if cfg.enable_routing:
            print(f"  Routing: ReMoE at step {cfg.route_start}, "
                  f"{cfg.n_clusters} clusters, "
                  f"target_sparsity={cfg.route_target_sparsity}")

    # Train
    print(f"\nTraining: steps {start_step} -> {cfg.total_steps}")
    model.train()
    t0 = time.time()

    for step in range(start_step, cfg.total_steps):
        x, y = loader.get_batch(device)

        # Depth sampling
        if cfg.use_poisson:
            plan = sample_depth(cfg.batch_size, cfg.mean_depth,
                                cfg.max_depth, cfg.bptt_depth, device)
        else:
            plan = sample_fixed(cfg.batch_size, cfg.mean_depth,
                                cfg.bptt_depth, device)

        # LR schedule
        lr = _lr_schedule(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = compiled_model(
                x, plan.total, plan.n_max, plan.k_max,
                labels=y, deterministic=False,
                outer_state=outer_state,
                step=step,
            )
            loss = out["loss"]

        loss.backward()

        # ─── Neural memory: update MLP weights (outside compiled forward) ───
        memory_loss = None
        if cfg.use_neural_memory and out.get("h_final_for_memory") is not None:
            if step >= cfg.memory_warmup_steps and step % cfg.memory_update_interval == 0:
                h_for_mem = out["h_final_for_memory"]
                for _ in range(cfg.memory_inner_steps):
                    memory_loss = model.neural_memory.update(
                        h_for_mem, return_stats=False,
                        differentiable=False,
                    )

        # ─── CMS: accumulate scores (BETWEEN backward and optimizer step) ───
        if prunable_mods and step <= cfg.prune_end:
            for _, _, mod in prunable_mods:
                mod.accumulate_scores()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ─── Re-zero dead tiles (optimizer momentum can resurrect them) ───
        if prune_round > 0 and not routing_active:
            for _, _, mod in prunable_mods:
                mod.rezero_dead()

        if cfg.use_outer_ssm and out.get("outer_state_out") is not None:
            outer_state = out["outer_state_out"].detach()

        # ─── Phase logic ─────────────────────────────────────────────────────
        if cfg.enable_pruning:
            # Gradual pruning
            if (prunable_mods and cfg.prune_start < step <= cfg.prune_end
                    and (step - cfg.prune_start) % cfg.prune_interval == 0):
                prune_round += 1
                total_killed = 0
                total_alive_before = 0
                for i, (ci, name, mod) in enumerate(prunable_mods):
                    killed, was_alive, density = mod.prune_fraction(cfg.prune_frac)
                    total_killed += killed
                    total_alive_before += was_alive
                current_density = sum(
                    m.density for _, _, m in prunable_mods
                ) / len(prunable_mods)
                print(f"  PRUNE R{prune_round} @ step {step}: killed {total_killed} tiles, "
                      f"density={current_density:.1%}")
                wandb.log({
                    "prune/round": prune_round,
                    "prune/density": current_density,
                    "prune/tiles_killed": total_killed,
                }, step=step)

            # Compaction at prune_end
            if step == cfg.prune_end and prune_round > 0:
                print(f"\n  ═══ COMPACTING at step {step} (density={current_density:.1%}) ═══")
                mem_pre = torch.cuda.memory_allocated() / 1e9

                # Frozen-dense compaction: zero dead tiles permanently, keep
                # dense weight. No shape change → no optimizer rebuild, no
                # recompilation. cuBLAS at this scale beats any sparse repr.
                n_clusters = cfg.n_clusters if cfg.enable_routing else None
                for ci, name, mod in prunable_mods:
                    K_new = mod.compact(n_clusters=n_clusters)
                    alive = mod.tile_mask.sum().item()
                    total = mod.R * mod.C
                    print(f"    core[{ci}].{name}: K_eff={K_new}, "
                          f"{alive}/{total} tiles alive ({alive/total:.1%})")

                mem_post = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM: {mem_pre:.1f}GB → {mem_post:.1f}GB "
                      f"(delta {mem_post - mem_pre:+.1f}GB)")
                print(f"  Optimizer + compiled graph PRESERVED (no shape change)")
                print(f"  ═══ COMPACTION COMPLETE ═══\n")

            # ReMoE routing activation
            if cfg.enable_routing and step == cfg.route_start and not routing_active:
                print(f"\n  ═══ ACTIVATING ReMoE ROUTING at step {step} ═══")
                # Import and create ReMoE routers for each core block's MLP
                from titans_core.routing.remoe_router import ReMoERouter

                for i, blk in enumerate(model.core):
                    mlp = blk.mlp
                    if not (hasattr(mlp, 'prunable') and mlp.prunable):
                        continue
                    gate_mod = mlp.w_gate
                    if not gate_mod._compacted:
                        print(f"  WARNING: core[{i}] not compacted, skipping routing")
                        continue
                    n_cl = gate_mod.n_clusters
                    d_in = cfg.d_model
                    router = ReMoERouter(d_in, n_cl, d_query=min(256, d_in),
                                         target_sparsity=cfg.route_target_sparsity)
                    router = router.to(device)
                    mlp.router = router
                    routed_blocks.append(mlp)

                # Add router params to optimizer
                if routed_blocks:
                    router_params = []
                    for rb in routed_blocks:
                        router_params.extend(list(rb.router.parameters()))
                    optimizer.add_param_group({
                        "params": router_params,
                        "lr": lr,
                        "weight_decay": 0.0,
                    })
                    n_rp = sum(p.numel() for p in router_params)
                    print(f"  Router params: {n_rp:,}")

                routing_active = True
                print(f"  ═══ ROUTING ACTIVE ═══\n")

        # Determine phase label
        if cfg.enable_pruning:
            if step <= cfg.prune_start:
                phase_label = "DENSE"
            elif step <= cfg.prune_end:
                phase_label = f"PRUNE R{prune_round}"
            elif step < getattr(cfg, 'route_start', cfg.total_steps + 1):
                phase_label = "SETTLE"
            elif routing_active:
                phase_label = "ROUTE"
            else:
                phase_label = "POST"
        else:
            phase_label = ""

        # Logging
        if step % LOG_INTERVAL == 0:
            loss_val = loss.item()
            ppl = math.exp(min(loss_val, 20.0))
            elapsed = time.time() - t0
            sps = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            eta_min = (cfg.total_steps - step) / max(sps, 0.01) / 60
            density_str = f" d={current_density:.2f}" if cfg.enable_pruning and current_density < 1.0 else ""
            phase_str = f" [{phase_label}]" if phase_label else ""
            if use_amp:
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step {step:6d}{phase_str} | loss {loss_val:.4f} | ppl {ppl:7.1f} | "
                      f"{sps:.1f} step/s | {mem_gb:.1f}GB{density_str} | eta {eta_min:.0f}m")
            else:
                print(f"  step {step:6d}{phase_str} | loss {loss_val:.4f} | ppl {ppl:7.1f} | "
                      f"{sps:.1f} step/s{density_str} | eta {eta_min:.0f}m")
            sys.stdout.flush()
            log_dict = {
                "train/loss": loss_val, "train/ppl": ppl,
                "train/lr": lr, "perf/steps_per_s": sps,
            }
            if cfg.enable_pruning:
                log_dict["train/density"] = current_density
            # Memory stats
            if cfg.use_neural_memory:
                if memory_loss is not None:
                    ml = memory_loss
                    log_dict["memory/loss"] = ml.item() if hasattr(ml, 'item') else ml
                if step % (LOG_INTERVAL * 5) == 0:
                    mstats = model.neural_memory.get_memory_stats()
                    for k, v in mstats.items():
                        log_dict[f"memory/{k}"] = v
            if cfg.use_mtp and out.get("mtp_loss") is not None:
                ml = out["mtp_loss"]
                log_dict["mtp/loss"] = ml.item() if hasattr(ml, 'item') else ml
            wandb.log(log_dict, step=max(step, 1))

        # Eval
        if step > 0 and step % EVAL_INTERVAL == 0:
            val_loss = evaluate(model, eval_batches, cfg, device, outer_state)
            val_ppl = math.exp(min(val_loss, 20.0))
            print(f"  EVAL step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
            wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

        # Checkpoint
        if step > 0 and step % CKPT_INTERVAL == 0:
            save_checkpoint(args.name, step, model, optimizer, outer_state,
                           prune_round=prune_round, current_density=current_density)

    # Final eval + save
    val_loss = evaluate(model, eval_batches, cfg, device, outer_state)
    val_ppl = math.exp(min(val_loss, 20.0))
    print(f"\n  Final: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}")
    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=cfg.total_steps)
    wandb.summary.update({"final_val_ppl": val_ppl, "final_val_loss": val_loss,
                          "n_params": n_params})

    save_checkpoint(args.name, cfg.total_steps, model, optimizer, outer_state,
                   prune_round=prune_round, current_density=current_density)
    wandb.finish()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="PyTorch ablation runner")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--overlay", default=None, help="Override config YAML")
    parser.add_argument("--name", required=True, help="Run name (also checkpoint dir)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    train(load_config(args.config, args.overlay), args)


if __name__ == "__main__":
    main()
