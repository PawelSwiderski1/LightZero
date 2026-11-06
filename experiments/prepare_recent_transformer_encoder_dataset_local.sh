#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE_DIR="${BASE_DIR:-.local_runs/branching_rooms_action_novelty_seed0}"
OUTPUT_DIR="${OUTPUT_DIR:-.local_runs/branching_rooms_action_novelty_macro_augmented_seed0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the project environment's Python executable." >&2
    exit 2
fi

"${PYTHON_BIN}" -u experiments/run_branching_rooms_action_novelty.py \
    --mode collect \
    --seed 0 \
    --device cpu \
    --output-dir "${BASE_DIR}"

"${PYTHON_BIN}" -u experiments/prepare_branching_rooms_macro_augmented_dataset.py \
    --base-dataset "${BASE_DIR}/experienced_trajectories.npz" \
    --output-dir "${OUTPUT_DIR}"

echo "Prepared ${OUTPUT_DIR}/experienced_trajectories.npz"
