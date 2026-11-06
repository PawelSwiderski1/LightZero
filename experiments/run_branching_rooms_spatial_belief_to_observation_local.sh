#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1

ONLINE_UPDATES_PER_COLLECT=${ONLINE_UPDATES_PER_COLLECT:-64}
ONLINE_ACTIVATION_UPDATES=${ONLINE_ACTIVATION_UPDATES:-64}
EXP_NAME=${EXP_NAME:-.local_runs/branching_rooms_spatial_belief_to_observation_64updates_local_seed0}

if [[ -e "${EXP_NAME}" ]]; then
    echo "Output already exists: ${EXP_NAME}" >&2
    echo "Remove it or set EXP_NAME to a new directory." >&2
    exit 1
fi

exec .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py \
    --condition shared_log_auxiliary_spatial_tree_online \
    --intrinsic-target bootstrapped_20step \
    --seed 0 \
    --exp-name "${EXP_NAME}" \
    --max-env-step 512 \
    --max-train-iter 256 \
    --episode-max-step 128 \
    --num-simulations 15 \
    --collector-env-num 1 \
    --n-episode 1 \
    --batch-size 8 \
    --replay-ratio 0.25 \
    --rnd-fresh-update-batch-size 8 \
    --spatial-tree-max-depth 5 \
    --spatial-tree-reward-mode gate \
    --spatial-tree-rnd-estimator head \
    --spatial-tree-memory-key-mode belief_to_observation \
    --spatial-tree-online-rollout-reconstruction-weight 1 \
    --spatial-tree-online-rollout-contrastive-weight 1 \
    --spatial-tree-online-belief-observation-alignment-weight 1 \
    --spatial-tree-online-predictive-equivalence-weight 0 \
    --spatial-tree-online-updates-per-collect "${ONLINE_UPDATES_PER_COLLECT}" \
    --spatial-tree-online-activation-updates "${ONLINE_ACTIVATION_UPDATES}" \
    --spatial-tree-online-batch-size 2 \
    --spatial-tree-online-roots-per-update 8 \
    --spatial-tree-online-replay-capacity 4 \
    --spatial-tree-online-evaluation-roots-per-episode 32 \
    --no-record-collect-video
