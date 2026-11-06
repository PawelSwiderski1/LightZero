#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKER="${REPO_ROOT}/cluster/slurm_branching_rooms_spatial_world_model.sbatch"
SOURCE_DATASET="${SOURCE_DATASET:-${REPO_ROOT}/.local_runs/branching_rooms_action_novelty_macro_augmented_seed0/experienced_trajectories.npz}"
TARGET_DATASET="${TARGET_DATASET:-${REPO_ROOT}/.local_runs/branching_rooms_muzero_spatial_dataset_seed17/experienced_trajectories.npz}"
SOURCE_LABEL="${SOURCE_LABEL:-br_spatial_key_long_source_s0}"
ADAPT_LABEL="${ADAPT_LABEL:-br_spatial_key_long_muzero_s17}"
SOURCE_TIME_LIMIT="${SOURCE_TIME_LIMIT:-06:00:00}"
ADAPT_TIME_LIMIT="${ADAPT_TIME_LIMIT:-04:00:00}"
RUN_ROOT="${LIGHTZERO_RUN_ROOT:-${SCRATCH:-/scratch/${USER}}/lightzero_runs}"

for path in "${WORKER}" "${SOURCE_DATASET}" "${TARGET_DATASET}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 2
    fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "source: sbatch --job-name=${SOURCE_LABEL} --time=${SOURCE_TIME_LIMIT} ${WORKER} source"
    echo "adapt:  sbatch --dependency=afterok:<source_job_id> --job-name=${ADAPT_LABEL} --time=${ADAPT_TIME_LIMIT} ${WORKER} adapt ${RUN_ROOT}/${SOURCE_LABEL}_<source_job_id>/best.pt"
    exit 0
fi

SOURCE_JOB_ID=$(
    RUN_LABEL="${SOURCE_LABEL}" \
    SOURCE_DATASET="${SOURCE_DATASET}" \
    TARGET_DATASET="${TARGET_DATASET}" \
        sbatch --parsable \
        --job-name="${SOURCE_LABEL}" \
        --time="${SOURCE_TIME_LIMIT}" \
        "${WORKER}" source
)
SOURCE_JOB_ID="${SOURCE_JOB_ID%%;*}"
SOURCE_CHECKPOINT="${RUN_ROOT}/${SOURCE_LABEL}_${SOURCE_JOB_ID}/best.pt"

ADAPT_JOB_ID=$(
    RUN_LABEL="${ADAPT_LABEL}" \
    SOURCE_DATASET="${SOURCE_DATASET}" \
    TARGET_DATASET="${TARGET_DATASET}" \
        sbatch --parsable \
        --dependency="afterok:${SOURCE_JOB_ID}" \
        --job-name="${ADAPT_LABEL}" \
        --time="${ADAPT_TIME_LIMIT}" \
        "${WORKER}" adapt "${SOURCE_CHECKPOINT}"
)
ADAPT_JOB_ID="${ADAPT_JOB_ID%%;*}"

echo "Submitted source job ${SOURCE_JOB_ID}"
echo "Submitted dependent MuZero-adaptation job ${ADAPT_JOB_ID}"
echo "Source output: ${RUN_ROOT}/${SOURCE_LABEL}_${SOURCE_JOB_ID}"
echo "Adapted output: ${RUN_ROOT}/${ADAPT_LABEL}_${ADAPT_JOB_ID}"
