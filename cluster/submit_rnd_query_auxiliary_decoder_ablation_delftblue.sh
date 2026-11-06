#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEEDS="${SEEDS:-0 1 2}"
MAX_ENV_STEP="${MAX_ENV_STEP:-45000}"
MAX_TRAIN_ITER="${MAX_TRAIN_ITER:-15000}"
BATCH="${BATCH:-all}"
CONFIGS="${CONFIGS:-}"
EXPERIMENT="experiments/run_mountain_car_rnd_query_auxiliary_only.py"

if [[ -n "${CONFIGS}" ]]; then
    CELLS=()
    for config in ${CONFIGS}; do
        case "${config}" in
            shared_log) CELLS+=("shared_log:shared:log") ;;
            member_log) CELLS+=("member_log:member:log") ;;
            shared_raw) CELLS+=("shared_raw:shared:raw") ;;
            member_raw) CELLS+=("member_raw:member:raw") ;;
            shared_normalized_raw) CELLS+=("shared_normalized_raw:shared:normalized_raw") ;;
            member_normalized_raw) CELLS+=("member_normalized_raw:member:normalized_raw") ;;
            *)
                echo "Unknown CONFIGS entry: ${config}" >&2
                exit 2
                ;;
        esac
    done
else
    case "${BATCH}" in
        1)
            CELLS=("shared_log:shared:log" "member_log:member:log")
            ;;
        2)
            CELLS=("shared_raw:shared:raw" "member_raw:member:raw")
            ;;
        3)
            CELLS=(
                "shared_normalized_raw:shared:normalized_raw"
                "member_normalized_raw:member:normalized_raw"
            )
            ;;
        all)
            CELLS=(
                "shared_log:shared:log"
                "member_log:member:log"
                "shared_raw:shared:raw"
                "member_raw:member:raw"
            )
            ;;
        *)
            echo "BATCH must be 1, 2, 3, or all; got ${BATCH}" >&2
            exit 2
            ;;
    esac
fi

for cell in "${CELLS[@]}"; do
    IFS=: read -r label reader_mode target_transform <<<"${cell}"
    if [[ "${reader_mode}" == "member" ]]; then
        reader_flag="--rnd-landscape-query-member-specific-reader"
    else
        reader_flag="--no-rnd-landscape-query-member-specific-reader"
    fi
    for seed in ${SEEDS}; do
        run_label="mcar_rnd_aux_decoder_${label}_seed${seed}"
        echo "Submitting ${run_label}"
        SEED="${seed}" RUN_LABEL="${run_label}" \
            "${REPO_ROOT}/cluster/submit_delftblue.sh" \
            "${EXPERIMENT}" \
            --max-env-step "${MAX_ENV_STEP}" \
            --max-train-iter "${MAX_TRAIN_ITER}" \
            "${reader_flag}" \
            --rnd-landscape-query-target-transform "${target_transform}" \
            --rnd-landscape-query-normalized-raw-reference-size 8192 \
            --rnd-landscape-query-normalized-raw-clip 10 \
            --rnd-landscape-query-normalized-raw-min-scale 1e-8 \
            --rnd-landscape-query-gradient-diagnostics-interval 100
    done
done
