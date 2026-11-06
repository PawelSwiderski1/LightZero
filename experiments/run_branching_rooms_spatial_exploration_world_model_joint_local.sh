#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DATASET="${DATASET:-${REPO_ROOT}/.local_runs/branching_rooms_action_novelty_macro_augmented_seed0/experienced_trajectories.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/.local_runs/branching_rooms_spatial_world_model_joint_key_matched_seed0}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_spatial_exploration_world_model.py" \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 0 \
    --horizon 5 \
    --autoencoder-epochs 80 \
    --dynamics-epochs 600 \
    --patience 80 \
    --batch-size 8 \
    --evaluation-batch-size 4 \
    --roots-per-episode 32 \
    --evaluation-roots-per-episode 0 \
    --representation-mode spatial_reconstruction \
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
    --device "${DEVICE}" \
    "$@" 2>&1 | tee -a "${OUTPUT_DIR}.log"
