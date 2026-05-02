#!/bin/bash
# Neural memory 15k sensitivity runs — sequential
# Alpha sweep + append+residual combo
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

# 1. Residual baseline (alpha=[0.0001, 0.003] — same as 100k run)
run_ablation "mem_res_15k" "configs/ablations/mem_residual.yaml"

# 2. Low alpha (longer memory, slower forgetting)
run_ablation "mem_res_alpha_low" "configs/ablations/mem_residual_alpha_low.yaml"

# 3. High alpha (shorter memory, faster forgetting)
run_ablation "mem_res_alpha_high" "configs/ablations/mem_residual_alpha_high.yaml"

# 4. Append + Residual combo
run_ablation "mem_append_res" "configs/ablations/mem_append_residual.yaml"

# 5. Pure append (for comparison)
run_ablation "mem_append_15k" "configs/ablations/mem_append.yaml"

echo "All sensitivity runs complete!"
