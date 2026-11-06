#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
MAX_ENV_STEP="${MAX_ENV_STEP:-45000}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-15000}"

EXPERIMENTS=(
    "baseline:experiments/run_mountain_car_full_intrinsic_rnd.py"
    "no_context:experiments/run_mountain_car_rnd_landscape_query_no_context.py"
    "detached_world:experiments/run_mountain_car_rnd_landscape_query_detached_world_model.py"
    "joint_query:experiments/run_mountain_car_rnd_landscape_query.py"
)

for item in "${EXPERIMENTS[@]}"; do
    label="${item%%:*}"
    experiment="${item#*:}"
    for seed in ${SEEDS}; do
        run_label="mcar_rnd_query_${label}_seed${seed}"
        echo "Submitting ${run_label}"
        SEED="${seed}" RUN_LABEL="${run_label}" \
            "${REPO_ROOT}/cluster/submit_delftblue.sh" \
            "${experiment}" \
            --max-env-step "${MAX_ENV_STEP}" \
            --max-train-iter "${MAX_TRAIN_ITER}"
    done
done
