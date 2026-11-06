#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEED="${SEED:-0}"
MAX_ENV_STEP="${MAX_ENV_STEP:-28800}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-7200}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-25}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
RUN_LABEL="branching_rooms_world_model_mature_shared_log_auxiliary_seed${SEED}"

echo "Submitting ${RUN_LABEL}"
SEED="${SEED}" RUN_LABEL="${RUN_LABEL}" TIME_LIMIT="${TIME_LIMIT}" \
    "${REPO_ROOT}/cluster/submit_delftblue.sh" \
    experiments/run_branching_rooms_intrinsic_rnd.py \
    --condition shared_log_auxiliary \
    --intrinsic-target bootstrapped_20step \
    --max-env-step "${MAX_ENV_STEP}" \
    --max-train-iter "${MAX_TRAIN_ITER}" \
    --num-simulations "${NUM_SIMULATIONS}"
