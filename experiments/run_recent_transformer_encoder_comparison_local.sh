#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATASET="${DATASET:-.local_runs/branching_rooms_action_novelty_macro_augmented_seed0/experienced_trajectories.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-.local_runs/recent_transformer_encoder_comparison_1024}"
BUDGETS="${BUDGETS:-256,512,1024}"
SEEDS="${SEEDS:-0 1 2}"
ENCODERS="${ENCODERS:-pooled_2x2 rgb_tile_7x7 symbolic_7x7}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the project environment's Python executable." >&2
    exit 2
fi
if [[ ! -f "${DATASET}" ]]; then
    echo "Dataset not found: ${DATASET}" >&2
    echo "Run experiments/prepare_recent_transformer_encoder_dataset_local.sh first." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
for encoder in ${ENCODERS}; do
    for seed in ${SEEDS}; do
        output="${OUTPUT_DIR}/${encoder}_seed${seed}.json"
        command=(
            "${PYTHON_BIN}" -u experiments/diagnose_recent_history_transformer.py
            --dataset "${DATASET}"
            --budgets "${BUDGETS}"
            --seed "${seed}"
            --observation-encoder "${encoder}"
            --output-json "${output}"
        )
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            echo "Running encoder=${encoder} seed=${seed}"
            "${command[@]}"
        fi
    done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    echo "Wrote comparison reports to ${OUTPUT_DIR}"
fi
