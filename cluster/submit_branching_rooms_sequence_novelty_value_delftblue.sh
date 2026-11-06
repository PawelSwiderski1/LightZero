#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
CONDITIONS="${CONDITIONS:-shared_log_auxiliary_sequence_novelty shared_log_auxiliary_sequence_novelty_value}"
MAX_ENV_STEP="${MAX_ENV_STEP:-28800}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-7200}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-25}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
EXPERIMENT="experiments/run_branching_rooms_intrinsic_rnd.py"

for condition in ${CONDITIONS}; do
    case "${condition}" in
        shared_log_auxiliary_sequence_novelty)
            condition_label="sequence_novelty_edge"
            ;;
        shared_log_auxiliary_sequence_novelty_value)
            condition_label="sequence_novelty_value"
            ;;
        *)
            echo "Unknown sequence-novelty condition: ${condition}" >&2
            exit 2
            ;;
    esac

    for seed in ${SEEDS}; do
        run_label="branching_rooms_long_bootstrap_${condition_label}_seed${seed}"
        echo "Submitting ${run_label}"
        SEED="${seed}" RUN_LABEL="${run_label}" TIME_LIMIT="${TIME_LIMIT}" \
            "${REPO_ROOT}/cluster/submit_delftblue.sh" \
            "${EXPERIMENT}" \
            --condition "${condition}" \
            --intrinsic-target bootstrapped_20step \
            --max-env-step "${MAX_ENV_STEP}" \
            --max-train-iter "${MAX_TRAIN_ITER}" \
            --num-simulations "${NUM_SIMULATIONS}" \
            --intrinsic-tree-leaf-value-weight 1 \
            --episodic-sequence-updates-per-collect 64 \
            --episodic-sequence-train-batch-size 1 \
            --episodic-sequence-roots-per-update 32
    done
done
