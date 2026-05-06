#!/bin/bash
set -e
cd "$(dirname "$0")"

STEPS=12000
BS=8
SL=2048
PYTHON=/home/wolfe/.venv/bin/python

echo "=== Attention Optimization Ablation ==="
echo "Steps: $STEPS | Batch: $BS | SeqLen: $SL"
echo "Started: $(date)"
echo ""

run_experiment() {
    local name=$1 attn=$2 kwargs=${3:-"{}"}
    echo "─── [$name] attn=$attn kwargs=$kwargs ───"
    echo "  Start: $(date)"
    $PYTHON train.py --attn "$attn" --name "$name" \
        --steps $STEPS --batch_size $BS --seq_len $SL \
        --attn_kwargs "$kwargs"
    echo "  Done:  $(date)"
    echo ""
}

# 1. Dense baseline (for fair comparison at these settings)
run_experiment "opt_dense_baseline" "dense"

# 2. GQA (n_kv_heads=4, 3:1 ratio)
run_experiment "opt_gqa" "gqa" '{"n_kv_heads": 4}'

# 3. QK-Norm (Gemma 2 style)
run_experiment "opt_qknorm" "qk_norm"

# 4. MLA (kv_rank=256, 3x KV compression)
run_experiment "opt_mla" "mla" '{"kv_rank": 256}'

# 5. XSA (exclude self-token)
run_experiment "opt_xsa" "xsa"

# 6. Residual Attention (alpha=0.1)
run_experiment "opt_residual" "residual_attn"

# 7. CoPE (Clipped RoPE, context_len matches training seq_len)
run_experiment "opt_cope" "dense_cope" "{\"context_len\": $SL}"

# 8. Combined (GQA + QK-Norm + Residual + CoPE)
run_experiment "opt_combined" "dense_opt" "{\"n_kv_heads\": 4, \"context_len\": $SL}"

echo "=== All ablations complete ==="
echo "Finished: $(date)"
