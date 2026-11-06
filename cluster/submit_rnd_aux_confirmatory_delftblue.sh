#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
MAX_ENV_STEP="${MAX_ENV_STEP:-45000}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-15000}"
BATCH="${BATCH:-1}"
CONFIGS="${CONFIGS:-}"
EXPERIMENT="experiments/run_mountain_car_rnd_aux_confirmatory_ablation.py"

if [[ -n "${CONFIGS}" ]]; then
    read -r -a CONDITIONS <<<"${CONFIGS}"
else
    case "${BATCH}" in
        1) CONDITIONS=(base joint_train_only) ;;
        2) CONDITIONS=(detached_train_only ordinary_matched_train_only) ;;
        3) CONDITIONS=(ordinary_exact_train_only joint_tree) ;;
        4) CONDITIONS=(joint_tree_no_private_reward) ;;
        all)
            CONDITIONS=(
                base
                joint_train_only
                detached_train_only
                ordinary_matched_train_only
                ordinary_exact_train_only
                joint_tree
                joint_tree_no_private_reward
            )
            ;;
        *)
            echo "BATCH must be 1, 2, 3, 4, or all; got ${BATCH}" >&2
            exit 2
            ;;
    esac
fi

for condition in "${CONDITIONS[@]}"; do
    case "${condition}" in
        base|joint_train_only|detached_train_only|ordinary_matched_train_only|ordinary_exact_train_only|joint_tree|joint_tree_no_private_reward) ;;
        *)
            echo "Unknown confirmatory condition: ${condition}" >&2
            exit 2
            ;;
    esac
    for seed in ${SEEDS}; do
        run_label="mcar_rnd_aux_confirm_${condition}_seed${seed}"
        echo "Submitting ${run_label}"
        SEED="${seed}" RUN_LABEL="${run_label}" \
            "${REPO_ROOT}/cluster/submit_delftblue.sh" \
            "${EXPERIMENT}" \
            --max-env-step "${MAX_ENV_STEP}" \
            --max-train-iter "${MAX_TRAIN_ITER}" \
            --confirmatory-condition "${condition}"
    done
done
