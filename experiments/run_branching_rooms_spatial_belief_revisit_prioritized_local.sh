#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1

ONLINE_UPDATES_PER_COLLECT=${ONLINE_UPDATES_PER_COLLECT:-256}
ONLINE_ACTIVATION_UPDATES=${ONLINE_ACTIVATION_UPDATES:-256}
PRIORITIZED_REPLAY_FRACTION=${PRIORITIZED_REPLAY_FRACTION:-0.5}
PRIORITIZED_REPLAY_ALPHA=${PRIORITIZED_REPLAY_ALPHA:-0.6}
PRIORITIZED_REPLAY_PRIORITY_MODE=${PRIORITIZED_REPLAY_PRIORITY_MODE:-belief_error}
BELIEF_DYNAMICS_CONSISTENCY_WEIGHT=${BELIEF_DYNAMICS_CONSISTENCY_WEIGHT:-0}
ONLINE_HISTORY_WINDOW=${ONLINE_HISTORY_WINDOW:-32}
MAX_ENV_STEP=${MAX_ENV_STEP:-512}
MAX_TRAIN_ITER=${MAX_TRAIN_ITER:-256}
EPISODE_MAX_STEP=${EPISODE_MAX_STEP:-256}
NUM_SIMULATIONS=${NUM_SIMULATIONS:-15}
EXP_NAME=${EXP_NAME:-.local_runs/branching_rooms_spatial_belief_revisit_prioritized_local_seed0}

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
    --max-env-step "${MAX_ENV_STEP}" \
    --max-train-iter "${MAX_TRAIN_ITER}" \
    --episode-max-step "${EPISODE_MAX_STEP}" \
    --num-simulations "${NUM_SIMULATIONS}" \
    --collector-env-num 1 \
    --n-episode 1 \
    --batch-size 8 \
    --replay-ratio 0.25 \
    --rnd-fresh-update-batch-size 8 \
    --spatial-tree-max-depth 5 \
    --spatial-tree-reward-mode gate \
    --spatial-tree-rnd-estimator head \
    --spatial-tree-memory-key-mode belief_revisit \
    --spatial-tree-online-revisit-history-scope episode \
    --spatial-tree-online-rollout-reconstruction-weight 1 \
    --spatial-tree-online-rollout-contrastive-weight 0 \
    --spatial-tree-online-observation-key-consistency-weight 0 \
    --spatial-tree-online-belief-dynamics-consistency-weight \
        "${BELIEF_DYNAMICS_CONSISTENCY_WEIGHT}" \
    --spatial-tree-online-belief-observation-alignment-weight 0 \
    --spatial-tree-online-pairwise-revisit-weight 1 \
    --spatial-tree-online-pairwise-revisit-negative-samples 8 \
    --spatial-tree-online-history-window "${ONLINE_HISTORY_WINDOW}" \
    --spatial-tree-online-prioritized-replay-fraction \
        "${PRIORITIZED_REPLAY_FRACTION}" \
    --spatial-tree-online-prioritized-replay-alpha \
        "${PRIORITIZED_REPLAY_ALPHA}" \
    --spatial-tree-online-prioritized-replay-priority-mode \
        "${PRIORITIZED_REPLAY_PRIORITY_MODE}" \
    --spatial-tree-online-predictive-equivalence-weight 0 \
    --spatial-tree-online-updates-per-collect "${ONLINE_UPDATES_PER_COLLECT}" \
    --spatial-tree-online-activation-updates "${ONLINE_ACTIVATION_UPDATES}" \
    --spatial-tree-online-batch-size 2 \
    --spatial-tree-online-roots-per-update 8 \
    --spatial-tree-online-replay-capacity 4 \
    --spatial-tree-online-evaluation-roots-per-episode 32 \
    --no-record-collect-video
