"""Profile a single training step to find the neural memory bottleneck."""
import sys, yaml, torch, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interop.pt_model import (
    InteropConfig, LoopedTransformerPT, sample_depth,
)

# Load config
with open("configs/pipeline_70k.yaml") as f:
    raw = yaml.safe_load(f)
m = raw["model"]
cfg = InteropConfig(
    d_model=m["d_model"], n_heads=m["n_heads"], d_ff=m["d_ff"],
    n_prelude=m["n_prelude"], n_core=m["n_core"], n_coda=m["n_coda"],
    vocab_size=m["vocab_size"], max_seq_len=m["max_seq_len"],  # full size
    mean_depth=m["mean_depth"], max_depth=m["max_depth"],
    use_poisson=m.get("use_poisson", True),
    use_swiglu=m.get("use_swiglu", False),
    use_qk_norm=m.get("use_qk_norm", False),
    n_kv_heads=m.get("n_kv_heads"),
    use_xsa=m.get("use_xsa", False),
    use_neural_memory=True,
    n_memory_layers=m.get("n_memory_layers", 4),
    d_memory=m.get("d_memory", 1024),
    memory_mode=m.get("memory_mode", "residual"),
    memory_theta_lr=m.get("memory_theta_lr", 0.01),
    memory_alpha_min=m.get("memory_alpha_min", 0.001),
    memory_alpha_max=m.get("memory_alpha_max", 0.1),
    memory_surprise_scale=m.get("memory_surprise_scale", 5.0),
    memory_eta_fixed=m.get("memory_eta_fixed", 0.95),
    memory_inner_steps=m.get("memory_inner_steps", 1),
    memory_warmup_steps=0,  # force memory active immediately
    memory_ramp_steps=0,
    memory_update_interval=m.get("memory_update_interval", 5),
    enable_pruning=False,
    tile_size=16,
)

device = torch.device("cuda")
model = LoopedTransformerPT(cfg).to(device)
model.train()

# Warmup
print("Warming up...")
for i in range(5):
    x = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    y = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    plan = sample_depth(20, cfg.mean_depth, cfg.max_depth, None, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, plan.total, plan.n_max, plan.k_max, labels=y, step=5000+i)
        out["loss"].backward()
    model.zero_grad()
torch.cuda.synchronize()
print("Warmup done.\n")

# Profile with torch profiler
print("=== PROFILING WITH NEURAL MEMORY ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    for i in range(10):
        x = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
        y = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
        plan = sample_depth(20, cfg.mean_depth, cfg.max_depth, None, device)
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, plan.total, plan.n_max, plan.k_max, labels=y, step=5000+i)
            out["loss"].backward()
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# Now disable memory and profile again
print("\n=== PROFILING WITHOUT NEURAL MEMORY ===")
cfg2 = InteropConfig(
    d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff,
    n_prelude=cfg.n_prelude, n_core=cfg.n_core, n_coda=cfg.n_coda,
    vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len,
    mean_depth=cfg.mean_depth, max_depth=cfg.max_depth,
    use_poisson=cfg.use_poisson,
    use_swiglu=cfg.use_swiglu,
    use_qk_norm=cfg.use_qk_norm,
    n_kv_heads=cfg.n_kv_heads,
    use_xsa=cfg.use_xsa,
    use_neural_memory=False,  # disabled
    enable_pruning=False,
    tile_size=16,
)
model2 = LoopedTransformerPT(cfg2).to(device)
model2.train()

# Warmup model2
for i in range(5):
    x = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    y = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    plan = sample_depth(20, cfg.mean_depth, cfg.max_depth, None, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model2(x, plan.total, plan.n_max, plan.k_max, labels=y, step=5000+i)
        out["loss"].backward()
    model2.zero_grad()
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof2:
    for i in range(10):
        x = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
        y = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
        plan = sample_depth(20, cfg.mean_depth, cfg.max_depth, None, device)
        model2.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model2(x, plan.total, plan.n_max, plan.k_max, labels=y, step=5000+i)
            out["loss"].backward()
        torch.cuda.synchronize()

print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# Timing comparison
print("\n=== TIMING COMPARISON ===")
torch.cuda.synchronize()

t0 = time.time()
for i in range(50):
    x = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    y = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    plan = sample_depth(20, cfg.mean_depth, cfg.max_depth, None, device)
    model.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, plan.total, plan.n_max, plan.k_max, labels=y, step=5000+i)
        out["loss"].backward()
    torch.cuda.synchronize()
t_mem = (time.time() - t0) / 50

t0 = time.time()
for i in range(50):
    x = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    y = torch.randint(0, cfg.vocab_size, (20, 1024), device=device)
    plan = sample_depth(20, cfg.mean_depth, cfg.max_depth, None, device)
    model2.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model2(x, plan.total, plan.n_max, plan.k_max, labels=y, step=5000+i)
        out["loss"].backward()
    torch.cuda.synchronize()
t_nomem = (time.time() - t0) / 50

print(f"With memory:    {t_mem*1000:.1f} ms/step  ({1/t_mem:.1f} step/s)")
print(f"Without memory: {t_nomem*1000:.1f} ms/step  ({1/t_nomem:.1f} step/s)")
print(f"Memory overhead: {(t_mem - t_nomem)*1000:.1f} ms  ({(t_mem/t_nomem - 1)*100:.0f}%)")
