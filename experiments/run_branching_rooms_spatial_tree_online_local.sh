#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-.local_runs/branching_rooms_spatial_tree_online_seed0}"
DEVICE="${DEVICE:-cuda}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONUNBUFFERED=1

device_flag=--cuda
if [[ "${DEVICE}" == "cpu" ]]; then
    device_flag=--no-cuda
fi

"${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_intrinsic_rnd.py" \
    --condition shared_log_auxiliary_spatial_tree_online \
    --seed 0 \
    --exp-name "${OUTPUT_DIR}" \
    --max-env-step 1024 \
    --max-train-iter 512 \
    --episode-max-step 128 \
    --num-simulations 5 \
    --collector-env-num 1 \
    --n-episode 1 \
    --batch-size 16 \
    --replay-ratio 0.5 \
    --rnd-fresh-update-batch-size 16 \
    --spatial-tree-max-depth 5 \
    --spatial-tree-reward-mode gate \
    --spatial-tree-rnd-estimator head \
    --spatial-tree-memory-key-mode belief \
    --spatial-tree-online-rollout-reconstruction-weight 1 \
    --spatial-tree-online-updates-per-collect 16 \
    --spatial-tree-online-activation-updates 32 \
    --spatial-tree-online-batch-size 2 \
    --spatial-tree-online-roots-per-update 16 \
    --spatial-tree-online-replay-capacity 8 \
    "${device_flag}" \
    --no-record-collect-video \
    "$@" 2>&1 | tee "${OUTPUT_DIR}.log"
