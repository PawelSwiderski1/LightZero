#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-br_spatial_belief_consistency_long_online_s0_1006812}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/hpc_results/branching_rooms_spatial_belief_consistency_long_online_20261017/${RUN_NAME}}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/spatial_exploration_tree_online.pt}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Missing extracted spatial model: ${CHECKPOINT}" >&2
    echo "Fetch it first with:" >&2
    echo "  ./cluster/fetch_branching_rooms_spatial_belief_consistency_model.sh ${RUN_NAME}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

evaluate() {
    local dataset="$1"
    local output_name="$2"
    "${PYTHON}" -u experiments/evaluate_branching_rooms_spatial_online_checkpoint.py \
        --checkpoint "${CHECKPOINT}" \
        --dataset "${dataset}" \
        --output-dir "${RUN_ROOT}/${output_name}" \
        --split all \
        --horizon 5 \
        --roots-per-episode 0 \
        --root-batch-size 128 \
        --device "${DEVICE}"
}

evaluate \
    .local_runs/branching_rooms_training_layout_coverage/experienced_trajectories.npz \
    independent_eval_training_layout_coverage
evaluate \
    .local_runs/branching_rooms_heldout_layout_coverage/experienced_trajectories.npz \
    independent_eval_heldout_layout_coverage
