#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
MAX_ENV_STEP="${MAX_ENV_STEP:-28800}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-7200}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-25}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
ONLINE_UPDATES_PER_COLLECT="${ONLINE_UPDATES_PER_COLLECT:-256}"
ONLINE_ACTIVATION_UPDATES="${ONLINE_ACTIVATION_UPDATES:-512}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-4}"
ONLINE_ROOTS_PER_UPDATE="${ONLINE_ROOTS_PER_UPDATE:-32}"
ONLINE_REPLAY_CAPACITY="${ONLINE_REPLAY_CAPACITY:-32}"
ONLINE_EVALUATION_ROOTS="${ONLINE_EVALUATION_ROOTS:-128}"
EXPERIMENT="experiments/run_branching_rooms_intrinsic_rnd.py"

for seed in ${SEEDS}; do
    run_label="br_spatial_belief_online_s${seed}"
    command=(
        "${REPO_ROOT}/cluster/submit_delftblue.sh"
        "${EXPERIMENT}"
        --condition shared_log_auxiliary_spatial_tree_online
        --intrinsic-target bootstrapped_20step
        --max-env-step "${MAX_ENV_STEP}"
        --max-train-iter "${MAX_TRAIN_ITER}"
        --num-simulations "${NUM_SIMULATIONS}"
        --spatial-tree-max-depth 5
        --spatial-tree-reward-mode gate
        --spatial-tree-rnd-estimator head
        --spatial-tree-memory-key-mode belief
        --spatial-tree-online-rollout-reconstruction-weight 1
        --spatial-tree-online-updates-per-collect \
            "${ONLINE_UPDATES_PER_COLLECT}"
        --spatial-tree-online-activation-updates \
            "${ONLINE_ACTIVATION_UPDATES}"
        --spatial-tree-online-batch-size "${ONLINE_BATCH_SIZE}"
        --spatial-tree-online-roots-per-update "${ONLINE_ROOTS_PER_UPDATE}"
        --spatial-tree-online-replay-capacity "${ONLINE_REPLAY_CAPACITY}"
        --spatial-tree-online-evaluation-roots-per-episode \
            "${ONLINE_EVALUATION_ROOTS}"
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'SEED=%q RUN_LABEL=%q TIME_LIMIT=%q' \
            "${seed}" "${run_label}" "${TIME_LIMIT}"
        printf ' %q' "${command[@]}"
        printf '\n'
        continue
    fi
    echo "Submitting ${run_label}"
    SEED="${seed}" RUN_LABEL="${run_label}" TIME_LIMIT="${TIME_LIMIT}" \
        "${command[@]}"
done
