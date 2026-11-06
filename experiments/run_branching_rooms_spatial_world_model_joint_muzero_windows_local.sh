#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/.local_runs/branching_rooms_muzero_spatial_dataset_seed17/experienced_trajectories.npz}"
WINDOW_DIR="${WINDOW_DIR:-${REPO_ROOT}/.local_runs/branching_rooms_muzero_spatial_dataset_seed17_window128_stride32}"
WINDOW_DATASET="${WINDOW_DATASET:-${WINDOW_DIR}/experienced_trajectories.npz}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/.local_runs/branching_rooms_spatial_world_model_joint_key_matched_seed0/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/.local_runs/branching_rooms_spatial_world_model_joint_muzero_windows_seed17}"
DEVICE="${DEVICE:-cuda}"

for path in "${PYTHON}" "${SOURCE_DATASET}" "${CHECKPOINT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

mkdir -p "${WINDOW_DIR}" "${OUTPUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -u "${REPO_ROOT}/experiments/window_branching_rooms_trajectory_dataset.py" \
    --input "${SOURCE_DATASET}" \
    --output "${WINDOW_DATASET}" \
    --window-actions 128 \
    --stride 32 \
    --horizon 5

"${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_spatial_exploration_world_model.py" \
    --dataset "${WINDOW_DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume-checkpoint "${CHECKPOINT}" \
    --seed 17 \
    --horizon 5 \
    --dynamics-epochs 700 \
    --patience 60 \
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
    --memory-novelty-weight 0 \
    --selection-metric key_novelty \
    --joint-representation-dynamics \
    --fresh-optimizer-on-resume \
    --reset-selection-on-resume \
    --evaluate-resume-before-training \
    --figure-count 4 \
    --device "${DEVICE}" \
    "$@" 2>&1 | tee -a "${OUTPUT_DIR}.log"
