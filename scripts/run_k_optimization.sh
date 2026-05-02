#!/bin/bash
# K inner step optimization — find minimum K that maintains quality
set -e

PYTHON="/home/wolfe/.venv/bin/python"
BASE="configs/ablation_15k_mem.yaml"
SCRIPT="scripts/run_ablation_pt.py"

run_ablation() {
    local name=$1
    local overlay=$2
    echo "=== Starting: $name ==="
    rm -f "checkpoints/ablation/$name/wandb_id.txt" "checkpoints/ablation/$name/ckpt.pt"
    $PYTHON -u $SCRIPT --config $BASE --overlay "$overlay" --name "$name" 2>&1 | tee "experiments/${name}.log"
    echo "=== Done: $name ==="
    echo ""
}

# K=1 (fastest — single inner gradient step)
run_ablation "mem_res_k1" "configs/ablations/mem_residual_k1.yaml"

# K=2
run_ablation "mem_res_k2" "configs/ablations/mem_residual_k2.yaml"

# K=3
run_ablation "mem_res_k3" "configs/ablations/mem_residual_k3.yaml"

echo "All K optimization runs complete!"
