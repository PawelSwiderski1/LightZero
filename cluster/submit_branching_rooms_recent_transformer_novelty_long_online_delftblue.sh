#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0}"
MAX_ENV_STEP="${MAX_ENV_STEP:-86400}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-21600}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-25}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SEQUENCE_UPDATES_PER_COLLECT="${SEQUENCE_UPDATES_PER_COLLECT:-64}"
SEQUENCE_BATCH_SIZE="${SEQUENCE_BATCH_SIZE:-1}"
SEQUENCE_ROOTS_PER_UPDATE="${SEQUENCE_ROOTS_PER_UPDATE:-32}"
HISTORY_WINDOW="${HISTORY_WINDOW:-40}"
SEQUENCE_INFERENCE_MODE="${SEQUENCE_INFERENCE_MODE:-cached_suffix}"
EXPERIMENT="experiments/run_branching_rooms_intrinsic_rnd.py"

for seed in ${SEEDS}; do
    run_label="br_recent_transformer_novelty_long_online_s${seed}"
    command=(
        "${REPO_ROOT}/cluster/submit_delftblue.sh"
        "${EXPERIMENT}"
        --condition shared_log_auxiliary_recent_transformer_novelty
        --intrinsic-target bootstrapped_20step
        --max-env-step "${MAX_ENV_STEP}"
        --max-train-iter "${MAX_TRAIN_ITER}"
        --num-simulations "${NUM_SIMULATIONS}"
        --intrinsic-tree-leaf-value-weight 1
        --episodic-sequence-history-window "${HISTORY_WINDOW}"
        --episodic-sequence-inference-mode "${SEQUENCE_INFERENCE_MODE}"
        --episodic-sequence-updates-per-collect \
            "${SEQUENCE_UPDATES_PER_COLLECT}"
        --episodic-sequence-train-batch-size "${SEQUENCE_BATCH_SIZE}"
        --episodic-sequence-roots-per-update \
            "${SEQUENCE_ROOTS_PER_UPDATE}"
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
