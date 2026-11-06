#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RUN_DIRECTORY_NAME" >&2
    exit 2
fi

RUN_NAME="$1"
REMOTE="${REMOTE:-pmswiderski@login.delftblue.tudelft.nl}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/pmswiderski/lightzero_runs}"
REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}/hpc_results/branching_rooms_world_model_control_20261006}"
DEST="${DEST_ROOT}/${RUN_NAME}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
RETRY_DELAY="${RETRY_DELAY:-30}"

mkdir -p "${DEST}"

copied=0
for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    echo "${RUN_NAME}: transfer attempt ${attempt}/${MAX_ATTEMPTS}"
    if rsync -a \
        --partial \
        --append-verify \
        --timeout=180 \
        --info=progress2 \
        --include='/ckpt/' \
        --include='/ckpt/iteration_*.pth.tar' \
        --exclude='/ckpt/*' \
        --include='/diagnostics/***' \
        --include='/log/***' \
        --include='/total_config.py' \
        --include='/formatted_total_config.py' \
        --exclude='*' \
        "${REMOTE}:${REMOTE_ROOT}/${RUN_NAME}/" \
        "${DEST}/"; then
        copied=1
        break
    fi
    sleep "${RETRY_DELAY}"
done

if [[ ${copied} -ne 1 ]]; then
    echo "Transfer did not complete after ${MAX_ATTEMPTS} attempts." >&2
    exit 1
fi

mapfile -t checkpoints < <(
    find "${DEST}/ckpt" -maxdepth 1 -type f -name 'iteration_*.pth.tar' -printf '%f\n' |
        sort -V
)

if [[ ${#checkpoints[@]} -lt 2 ]]; then
    echo "Expected initialization and final iteration checkpoints under ${DEST}/ckpt." >&2
    printf 'Found: %s\n' "${checkpoints[*]:-none}" >&2
    exit 1
fi

for required in total_config.py diagnostics/environment_coverage_cells.csv; do
    if [[ ! -s "${DEST}/${required}" ]]; then
        echo "Missing required file after transfer: ${DEST}/${required}" >&2
        exit 1
    fi
done

echo "Fetched ${RUN_NAME} to ${DEST}"
printf 'Checkpoint: %s\n' "${checkpoints[@]}"
