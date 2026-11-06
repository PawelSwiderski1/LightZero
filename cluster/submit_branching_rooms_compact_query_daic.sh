#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
CONDITIONS="${CONDITIONS:-compact_query_projected_memory compact_query_raw_ssl compact_query_raw_memory}"
MAX_ENV_STEP="${MAX_ENV_STEP:-28800}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-7200}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-25}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-16GB}"
EXPERIMENT="experiments/run_branching_rooms_intrinsic_rnd.py"

for condition in ${CONDITIONS}; do
    case "${condition}" in
        compact_query|compact_query_no_context|compact_query_projected_memory|compact_query_raw_ssl|compact_query_raw_memory) ;;
        *)
            echo "Unknown compact-query condition: ${condition}" >&2
            exit 2
            ;;
    esac
    for seed in ${SEEDS}; do
        run_label="branching_rooms_long_bootstrap_${condition}_seed${seed}"
        echo "Submitting ${run_label}"
        SEED="${seed}" RUN_LABEL="${run_label}" TIME_LIMIT="${TIME_LIMIT}" \
            CPUS_PER_TASK="${CPUS_PER_TASK}" MEMORY="${MEMORY}" \
            "${REPO_ROOT}/cluster/submit_daic.sh" \
            "${EXPERIMENT}" \
            --condition "${condition}" \
            --intrinsic-target bootstrapped_20step \
            --max-env-step "${MAX_ENV_STEP}" \
            --max-train-iter "${MAX_TRAIN_ITER}" \
            --num-simulations "${NUM_SIMULATIONS}"
    done
done
