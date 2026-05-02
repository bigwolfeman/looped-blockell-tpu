#!/bin/bash
# Wait for MTP to finish, then run combined 15k ablation
set -e

PYTHON="/home/wolfe/.venv/bin/python"

echo "Waiting for MTP ablation to finish..."
while pgrep -f "run_ablation_pt.*mtp3_15k" > /dev/null 2>&1; do
    sleep 30
done
echo "MTP done. Starting combined 15k run."

rm -f checkpoints/ablation/full_stack_15k/wandb_id.txt checkpoints/ablation/full_stack_15k/ckpt.pt
$PYTHON -u scripts/run_ablation_pt.py \
    --config configs/ablation_15k_mem.yaml \
    --overlay configs/ablations/full_stack.yaml \
    --name full_stack_15k 2>&1 | tee experiments/full_stack_15k.log

echo "=== Full stack 15k complete ==="
