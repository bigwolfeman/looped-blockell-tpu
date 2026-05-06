"""Launch the top-4 priority ablation experiments sequentially.

Dense (ceiling) → Sliding Window (floor) → NSA (baseline) → NSA+MoSA (hypothesis)
All at seq_len=2048, 20k steps, with scaling measurement.
"""

import subprocess
import sys
import time

PYTHON = sys.executable
SEQ_LEN = 2048
STEPS = 20000

EXPERIMENTS = [
    {
        "name": "dense_baseline",
        "attn": "dense",
        "batch_size": 8,
        "extra_args": [],
    },
    {
        "name": "sliding_window_2k",
        "attn": "sliding_window",
        "batch_size": 8,
        "extra_args": ["--window_size", "2048"],
    },
    {
        "name": "nsa_faithful",
        "attn": "nsa",
        "batch_size": 4,
        "extra_args": [],
    },
    {
        "name": "nsa_mosa_hybrid",
        "attn": "nsa_mosa",
        "batch_size": 8,
        "extra_args": [],
    },
]


def run_experiment(exp: dict):
    cmd = [
        PYTHON, "train.py",
        "--attn", exp["attn"],
        "--name", exp["name"],
        "--seq_len", str(SEQ_LEN),
        "--steps", str(STEPS),
        "--batch_size", str(exp["batch_size"]),
        "--measure_scaling",
        *exp["extra_args"],
    ]
    print(f"\n{'='*60}")
    print(f"  Starting: {exp['name']} ({exp['attn']}, bs={exp['batch_size']})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    status = "✓ DONE" if result.returncode == 0 else f"✗ FAILED (rc={result.returncode})"
    print(f"\n  {exp['name']}: {status} in {elapsed/60:.1f}min\n")
    return result.returncode == 0


if __name__ == "__main__":
    results = {}
    for exp in EXPERIMENTS:
        ok = run_experiment(exp)
        results[exp["name"]] = ok
        if not ok:
            print(f"  WARNING: {exp['name']} failed, continuing with remaining experiments")

    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  Check wandb project: subq-attention-ablation")
