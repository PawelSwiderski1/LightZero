#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RUN_DIRECTORY" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(realpath "$1")"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
SAMPLES_PER_DEPTH="${SAMPLES_PER_DEPTH:-1024}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-256}"

mapfile -t checkpoints < <(
    find "${RUN_DIR}/ckpt" -maxdepth 1 -type f -name 'iteration_*.pth.tar' -printf '%f\n' |
        sort -V
)
if [[ ${#checkpoints[@]} -lt 2 ]]; then
    echo "Expected at least two iteration checkpoints under ${RUN_DIR}/ckpt." >&2
    exit 1
fi

initial_checkpoint="${checkpoints[0]}"
final_checkpoint="${checkpoints[${#checkpoints[@]} - 1]}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONUNBUFFERED=1

run_diagnostic() {
    local checkpoint="$1"
    local root_scope="$2"
    "${PYTHON}" -u "${REPO_ROOT}/experiments/diagnose_branching_rooms_world_model.py" \
        --run-dir "${RUN_DIR}" \
        --checkpoint "${checkpoint}" \
        --device "${DEVICE}" \
        --horizon 5 \
        --samples-per-depth "${SAMPLES_PER_DEPTH}" \
        --candidate-count "${CANDIDATE_COUNT}" \
        --root-scope "${root_scope}"
}

echo "Initialization control: ${initial_checkpoint}"
run_diagnostic "${initial_checkpoint}" all

echo "Mature checkpoint: ${final_checkpoint}"
run_diagnostic "${final_checkpoint}" all
run_diagnostic "${final_checkpoint}" visited
run_diagnostic "${final_checkpoint}" unvisited
