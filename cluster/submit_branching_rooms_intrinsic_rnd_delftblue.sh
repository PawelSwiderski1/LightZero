#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
CONDITIONS="${CONDITIONS:-baseline shared_log_auxiliary}"
MAX_ENV_STEP="${MAX_ENV_STEP:-30000}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-10000}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-25}"
EXPERIMENT="experiments/run_branching_rooms_intrinsic_rnd.py"

for condition in ${CONDITIONS}; do
    case "${condition}" in
        baseline|shared_log_auxiliary) ;;
        *)
            echo "Unknown BranchingRooms condition: ${condition}" >&2
            exit 2
            ;;
    esac
    for seed in ${SEEDS}; do
        run_label="branching_rooms_rnd_${condition}_seed${seed}"
        echo "Submitting ${run_label}"
        SEED="${seed}" RUN_LABEL="${run_label}" \
            "${REPO_ROOT}/cluster/submit_delftblue.sh" \
            "${EXPERIMENT}" \
            --condition "${condition}" \
            --max-env-step "${MAX_ENV_STEP}" \
            --max-train-iter "${MAX_TRAIN_ITER}" \
            --num-simulations "${NUM_SIMULATIONS}"
    done
done
