#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0}"
MAX_ENV_STEP="${MAX_ENV_STEP:-86400}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-21600}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-25}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
ONLINE_UPDATES_PER_COLLECT="${ONLINE_UPDATES_PER_COLLECT:-256}"
ONLINE_ACTIVATION_UPDATES="${ONLINE_ACTIVATION_UPDATES:-512}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-4}"
ONLINE_ROOTS_PER_UPDATE="${ONLINE_ROOTS_PER_UPDATE:-32}"
ONLINE_REPLAY_CAPACITY="${ONLINE_REPLAY_CAPACITY:-32}"
ONLINE_EVALUATION_ROOTS="${ONLINE_EVALUATION_ROOTS:-128}"
PAIRWISE_REVISIT_WEIGHT="${PAIRWISE_REVISIT_WEIGHT:-1.0}"
PAIRWISE_REVISIT_NEGATIVE_SAMPLES="${PAIRWISE_REVISIT_NEGATIVE_SAMPLES:-8}"
EXPERIMENT="experiments/run_branching_rooms_intrinsic_rnd.py"

for seed in ${SEEDS}; do
    run_label="br_spatial_belief_revisit_long_online_s${seed}"
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
        --spatial-tree-memory-key-mode belief_revisit
        --spatial-tree-online-rollout-reconstruction-weight 1
        --spatial-tree-online-rollout-contrastive-weight 0
        --spatial-tree-online-observation-key-consistency-weight 0
        --spatial-tree-online-belief-observation-alignment-weight 0
        --spatial-tree-online-pairwise-revisit-weight \
            "${PAIRWISE_REVISIT_WEIGHT}"
        --spatial-tree-online-pairwise-revisit-negative-samples \
            "${PAIRWISE_REVISIT_NEGATIVE_SAMPLES}"
        --spatial-tree-online-predictive-equivalence-weight 0
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
