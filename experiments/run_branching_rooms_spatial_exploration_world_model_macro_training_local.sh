#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/.local_runs/branching_rooms_action_novelty_macro_augmented_seed0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/.local_runs/branching_rooms_spatial_exploration_world_model_macro_training_seed0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -u "${REPO_ROOT}/experiments/prepare_branching_rooms_macro_augmented_dataset.py" \
    --base-dataset "${REPO_ROOT}/.local_runs/branching_rooms_action_novelty_seed0/experienced_trajectories.npz" \
    --output-dir "${DATA_DIR}"

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_spatial_exploration_world_model.py" \
    --dataset "${DATA_DIR}/experienced_trajectories.npz" \
    --output-dir "${OUTPUT_DIR}" \
    --device auto \
    --dynamics-epochs 160 \
    --patience 20 \
    "$@" 2>&1 | tee -a "${OUTPUT_DIR}.log"
