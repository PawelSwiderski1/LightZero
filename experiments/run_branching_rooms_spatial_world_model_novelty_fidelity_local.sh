#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DATASET="${DATASET:-${REPO_ROOT}/.local_runs/branching_rooms_muzero_spatial_dataset_seed17_window128_stride32/experienced_trajectories.npz}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/.local_runs/branching_rooms_spatial_world_model_joint_key_matched_seed0/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/.local_runs/branching_rooms_spatial_world_model_novelty_fidelity_seed17}"
DEVICE="${DEVICE:-cuda}"

for path in "${PYTHON}" "${DATASET}" "${CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_spatial_exploration_world_model.py" \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume-checkpoint "${CHECKPOINT}" \
    --seed 17 \
    --horizon 5 \
    --dynamics-epochs 650 \
    --patience 40 \
    --batch-size 8 \
    --evaluation-batch-size 4 \
    --roots-per-episode 32 \
    --evaluation-roots-per-episode 0 \
    --representation-mode spatial_reconstruction \
    --learning-rate 1e-4 \
    --direct-reconstruction-weight 2 \
    --direct-contrastive-weight 1 \
    --inverse-action-weight 0.1 \
    --variance-weight 0.1 \
    --latent-dynamics-weight 0 \
    --key-dynamics-weight 1 \
    --rollout-reconstruction-weight 0 \
    --rollout-contrastive-weight 1 \
    --memory-novelty-weight 10 \
    --selection-metric novelty_fidelity \
    --joint-representation-dynamics \
    --fresh-optimizer-on-resume \
    --evaluate-resume-before-training \
    --figure-count 4 \
    --device "${DEVICE}" \
    "$@" 2>&1 | tee -a "${OUTPUT_DIR}.log"
