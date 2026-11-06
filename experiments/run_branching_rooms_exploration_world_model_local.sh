#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/.local_runs/branching_rooms_exploration_world_model_two_stage_seed0}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_exploration_world_model.py" \
    --dataset "${REPO_ROOT}/.local_runs/branching_rooms_action_novelty_seed0/experienced_trajectories.npz" \
    --output-dir "${OUTPUT_DIR}" \
    --device auto \
    "$@" 2>&1 | tee "${OUTPUT_DIR}.log"
