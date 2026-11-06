#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-br_spatial_belief_revisit_long_online_s0_998715}"
CHECKPOINT_ITERATION="${CHECKPOINT_ITERATION:-21600}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/hpc_results/branching_rooms_spatial_belief_revisit_long_online_20261015/${RUN_NAME}}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/ckpt/iteration_${CHECKPOINT_ITERATION}.pth.tar}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    echo "Fetch it first with:" >&2
    echo "  ./cluster/fetch_branching_rooms_spatial_belief_revisit_long_checkpoint.sh ${RUN_NAME} ${CHECKPOINT_ITERATION}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -u experiments/evaluate_branching_rooms_spatial_online_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset .local_runs/branching_rooms_muzero_spatial_dataset_seed17/experienced_trajectories.npz \
    --output-dir "${RUN_ROOT}/independent_eval_muzero_seed17_all_roots" \
    --split all \
    --horizon 5 \
    --roots-per-episode 0 \
    --root-batch-size 128 \
    --device "${DEVICE}"

# This older dataset also changes the RGB rendering slightly. It is therefore a
# secondary observation-shift stress test rather than the primary trajectory test.
"${PYTHON}" -u experiments/evaluate_branching_rooms_spatial_online_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset .local_runs/branching_rooms_action_novelty_macro_augmented_seed0/experienced_trajectories.npz \
    --output-dir "${RUN_ROOT}/independent_eval_macro_render_shift" \
    --split heldout_policy \
    --horizon 5 \
    --roots-per-episode 128 \
    --root-batch-size 128 \
    --device "${DEVICE}"
