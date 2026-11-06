#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
MAX_ENV_STEP="${MAX_ENV_STEP:-45000}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-15000}"
BATCH="${BATCH:-1}"
CONFIGS="${CONFIGS:-}"
EXPERIMENT="experiments/run_mountain_car_intrinsic_reward_head_ablation.py"

if [[ -n "${CONFIGS}" ]]; then
    read -r -a CELLS <<<"${CONFIGS}"
else
    case "${BATCH}" in
        1) CELLS=(double_loss log_target) ;;
        2) CELLS=(shared_head) ;;
        3) CELLS=(shared_log) ;;
        all) CELLS=(double_loss log_target shared_head shared_log) ;;
        *)
            echo "BATCH must be 1, 2, 3, or all; got ${BATCH}" >&2
            exit 2
            ;;
    esac
fi

for ablation in "${CELLS[@]}"; do
    case "${ablation}" in
        double_loss|log_target|shared_head|shared_log) ;;
        *)
            echo "Unknown ablation: ${ablation}" >&2
            exit 2
            ;;
    esac
    for seed in ${SEEDS}; do
        run_label="mcar_rnd_reward_head_${ablation}_seed${seed}"
        echo "Submitting ${run_label}"
        SEED="${seed}" RUN_LABEL="${run_label}" \
            "${REPO_ROOT}/cluster/submit_delftblue.sh" \
            "${EXPERIMENT}" \
            --max-env-step "${MAX_ENV_STEP}" \
            --max-train-iter "${MAX_TRAIN_ITER}" \
            --reward-head-ablation "${ablation}"
    done
done
