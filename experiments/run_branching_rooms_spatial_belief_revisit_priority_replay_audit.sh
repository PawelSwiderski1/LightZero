#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-br_spatial_belief_revisit_priority_long_online_s0_1021540}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/hpc_results/branching_rooms_spatial_belief_revisit_priority_long_online_20261018/${RUN_NAME}}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/spatial_exploration_tree_online_with_replay.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/exact_replay_audit}"

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Missing replay-inclusive spatial checkpoint: ${CHECKPOINT}" >&2
    echo "Fetch it first with:" >&2
    echo "  INCLUDE_REPLAY=1 ./cluster/fetch_branching_rooms_spatial_belief_revisit_priority_model.sh ${RUN_NAME}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

arguments=(
    --checkpoint "${CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --horizon 5
    --roots-per-episode 0
    --root-batch-size 128
)
if [[ -n "${DEVICE:-}" ]]; then
    arguments+=(--device "${DEVICE}")
fi

"${PYTHON}" -u \
    experiments/evaluate_branching_rooms_spatial_replay_checkpoint.py \
    "${arguments[@]}"
