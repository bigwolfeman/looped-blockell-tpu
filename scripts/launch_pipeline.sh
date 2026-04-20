#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Looped Block-ELL: Multi-TPU Pipeline Launcher
#
# Runs the full Loop→Prune→Compact→Route pipeline across TPU node scaling:
#   Phase B: 3 TPU nodes (FSDP) — dense warmup + gradual pruning
#   Phase C: 1 TPU node — compact model + routing
#
# Usage:
#   ./scripts/launch_pipeline.sh --config configs/large.yaml \
#       --project my-gcp-project --zone us-central2-b \
#       --tpu-type v6e-8 --nodes 3 \
#       --gcs-bucket gs://my-bucket/checkpoints \
#       --wandb-run looped_large_pipeline
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────
CONFIG="configs/large.yaml"
GCP_PROJECT=""
GCP_ZONE="us-central2-b"
TPU_TYPE="v6e-8"
N_NODES=3
GCS_BUCKET=""
WANDB_RUN="looped_pipeline"
TPU_PREFIX="looped-blockell"
TPU_RUNTIME="tpu-ubuntu2204-base"
SEED=42

# ─── Parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2"; shift 2 ;;
        --project)      GCP_PROJECT="$2"; shift 2 ;;
        --zone)         GCP_ZONE="$2"; shift 2 ;;
        --tpu-type)     TPU_TYPE="$2"; shift 2 ;;
        --nodes)        N_NODES="$2"; shift 2 ;;
        --gcs-bucket)   GCS_BUCKET="$2"; shift 2 ;;
        --wandb-run)    WANDB_RUN="$2"; shift 2 ;;
        --tpu-prefix)   TPU_PREFIX="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$GCP_PROJECT" ]] && { echo "ERROR: --project required"; exit 1; }
[[ -z "$GCS_BUCKET" ]]  && { echo "ERROR: --gcs-bucket required"; exit 1; }

CKPT_DIR="${GCS_BUCKET}/${WANDB_RUN}"
COMPACT_CKPT="${CKPT_DIR}/compact"
SAFETY_CKPT="${CKPT_DIR}/pre_compact_safety"

echo "═══════════════════════════════════════════════════════════════"
echo "  Looped Block-ELL Pipeline"
echo "  Config:   ${CONFIG}"
echo "  TPU:      ${N_NODES}x ${TPU_TYPE}"
echo "  GCS:      ${CKPT_DIR}"
echo "  Wandb:    ${WANDB_RUN}"
echo "═══════════════════════════════════════════════════════════════"

# ─── Helper: create TPU VMs ───────────────────────────────────────────────
create_tpu_nodes() {
    local count=$1
    echo "Creating ${count} TPU nodes..."
    for i in $(seq 0 $((count - 1))); do
        local name="${TPU_PREFIX}-${i}"
        echo "  Creating ${name} (${TPU_TYPE})..."
        gcloud compute tpus tpu-vm create "${name}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            --accelerator-type="${TPU_TYPE}" \
            --version="${TPU_RUNTIME}" \
            --quiet &
    done
    wait
    echo "All ${count} nodes created."
}

# ─── Helper: delete TPU VMs ──────────────────────────────────────────────
delete_tpu_nodes() {
    local start=$1
    local end=$2
    echo "Deleting TPU nodes ${start}..${end}..."
    for i in $(seq "${start}" "${end}"); do
        local name="${TPU_PREFIX}-${i}"
        echo "  Deleting ${name}..."
        gcloud compute tpus tpu-vm delete "${name}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            --quiet &
    done
    wait
    echo "Nodes ${start}..${end} deleted."
}

# ─── Helper: run command on all TPU nodes ─────────────────────────────────
run_on_all() {
    local count=$1
    shift
    local cmd="$*"
    for i in $(seq 0 $((count - 1))); do
        local name="${TPU_PREFIX}-${i}"
        gcloud compute tpus tpu-vm ssh "${name}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            --command="${cmd}" &
    done
    wait
}

# ─── Helper: run training on TPU nodes ────────────────────────────────────
run_training() {
    local count=$1
    local phase=$2
    local extra_args=$3

    local train_cmd="cd ~/looped-blockell-tpu && \
        python scripts/train.py \
            --config ${CONFIG} \
            --wandb-run ${WANDB_RUN}_${phase} \
            --checkpoint-dir ${CKPT_DIR} \
            --seed ${SEED} \
            ${extra_args}"

    if [[ ${count} -eq 1 ]]; then
        # Single node: direct execution
        local name="${TPU_PREFIX}-0"
        echo "Running Phase ${phase} on ${name}..."
        gcloud compute tpus tpu-vm ssh "${name}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            --command="${train_cmd}"
    else
        # Multi-node: coordinate with JAX multi-host
        echo "Running Phase ${phase} on ${count} nodes..."
        for i in $(seq 0 $((count - 1))); do
            local name="${TPU_PREFIX}-${i}"
            local coordinator="${TPU_PREFIX}-0"
            local coord_addr
            coord_addr=$(gcloud compute tpus tpu-vm describe "${coordinator}" \
                --project="${GCP_PROJECT}" \
                --zone="${GCP_ZONE}" \
                --format="value(networkEndpoints[0].ipAddress)")

            local multi_cmd="cd ~/looped-blockell-tpu && \
                JAX_COORDINATOR_ADDRESS=${coord_addr}:1234 \
                JAX_COORDINATOR_PORT=1234 \
                JAX_NUM_PROCESSES=${count} \
                JAX_PROCESS_ID=${i} \
                ${train_cmd}"

            gcloud compute tpus tpu-vm ssh "${name}" \
                --project="${GCP_PROJECT}" \
                --zone="${GCP_ZONE}" \
                --command="${multi_cmd}" &
        done
        wait
    fi
}

# ─── Helper: deploy code to TPU nodes ────────────────────────────────────
deploy_code() {
    local count=$1
    echo "Deploying code to ${count} nodes..."
    for i in $(seq 0 $((count - 1))); do
        local name="${TPU_PREFIX}-${i}"
        # Copy code
        gcloud compute tpus tpu-vm scp --recurse \
            "$(dirname "$0")/.." "${name}:~/looped-blockell-tpu" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" &
    done
    wait

    # Install deps on all nodes
    run_on_all "${count}" "cd ~/looped-blockell-tpu && pip install -e '.[colab]' -q"
    echo "Code deployed and installed."
}

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE B: Multi-node training (dense → prune → compact checkpoint)
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE B: ${N_NODES}-node training (loop + prune)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Create nodes
create_tpu_nodes "${N_NODES}"

# Deploy code
deploy_code "${N_NODES}"

# Run Phase B (train.py exits after compaction checkpoint)
run_training "${N_NODES}" "phase_b" "--phase b --compact-checkpoint ${COMPACT_CKPT}"

echo ""
echo "Phase B complete. Compact checkpoint saved to ${COMPACT_CKPT}"

# ═══════════════════════════════════════════════════════════════════════════
#  TRANSITION: Scale down 3 → 1 node
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TRANSITION: Releasing ${N_NODES}-1 nodes                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
#
# Memory note: Phase B's train.py process terminated cleanly at prune_end
# (--phase b causes an explicit return after saving the compact checkpoint).
# Process termination guarantees full HBM reclamation:
#   - All JAX/XLA device buffers released
#   - XLA compiled-program cache cleared
#   - Python heap collected
# Phase C will start fresh on node 0 from the GCS compact checkpoint.
# No in-process memory migration occurs — the checkpoint is the only handoff.

if [[ ${N_NODES} -gt 1 ]]; then
    delete_tpu_nodes 1 $((N_NODES - 1))
    echo "Released $((N_NODES - 1)) nodes. Node 0 retained for Phase C."
else
    echo "Already on 1 node, no scaling needed."
fi

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE C: Single-node training (compact model + routing)
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PHASE C: 1-node training (compact + route)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

run_training 1 "phase_c" "--phase c --resume-from ${COMPACT_CKPT}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  Wandb: ${WANDB_RUN}_phase_b, ${WANDB_RUN}_phase_c"
echo "  Final checkpoint: ${CKPT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# Cleanup: delete remaining node
read -p "Delete remaining TPU node? [y/N] " confirm
if [[ "${confirm}" =~ ^[Yy]$ ]]; then
    delete_tpu_nodes 0 0
    echo "All TPU nodes deleted."
fi
