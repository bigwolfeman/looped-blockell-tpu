#!/bin/bash
# Run all PyTorch-compatible ablations sequentially.
# Each takes ~2.5h at 1.7 step/s for 15k steps. Total: ~12.5h.
set -e

cd "$(dirname "$0")/.."
PYTHON=/home/wolfe/.venv/bin/python
BASE=configs/ablation.yaml

echo "=== Starting ablation suite ==="
echo "Config: 3p+6c+3coda, d=512, 63M params, 15k steps each"
echo ""

# 1. Baseline (no overlays)
echo ">>> [1/5] baseline"
$PYTHON scripts/run_ablation_pt.py --config $BASE --name pt_baseline

# 2. Outer SSM (detached — no gradient through cross-sequence state)
echo ">>> [2/5] outer_ssm_detach"
$PYTHON scripts/run_ablation_pt.py --config $BASE --overlay configs/ablations/outer_ssm_detach.yaml --name pt_outer_ssm_detach

# 3. Outer SSM (with gradient — cross-sequence BPTT)
echo ">>> [3/5] outer_ssm_grad"
$PYTHON scripts/run_ablation_pt.py --config $BASE --overlay configs/ablations/outer_ssm_grad.yaml --name pt_outer_ssm_grad

# 4. Loop-boundary hyper-connections
echo ">>> [4/5] loop_boundary_hc"
$PYTHON scripts/run_ablation_pt.py --config $BASE --overlay configs/ablations/loop_boundary_hc.yaml --name pt_loop_boundary_hc

# 5. No-loop benchmark (T=1)
echo ">>> [5/5] no_loop_benchmark"
$PYTHON scripts/run_ablation_pt.py --config $BASE --overlay configs/ablations/no_loop_benchmark.yaml --name pt_no_loop

echo ""
echo "=== All ablations complete ==="
