#!/usr/bin/env bash

set -euo pipefail

ACCOUNT="${ACCOUNT:-ewi-insy-prb}"
PARTITION="${PARTITION:-insy}"
QOS="${QOS:-short}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-4GB}"
GRES="${GRES:-gpu:a40:1}"

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="${REPO_ROOT}/cluster/slurm_daic_a40.sbatch"

mkdir -p "${REPO_ROOT}/slurm_logs"

SBATCH_ARGS=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --time="${TIME_LIMIT}"
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --gres="${GRES}"
    --mem="${MEMORY}"
)

if [[ -n "${RUN_LABEL:-}" ]]; then
    SBATCH_ARGS+=(--job-name="${RUN_LABEL}")
fi

echo "Submitting to DAIC ${PARTITION}: ${ACCOUNT}, QOS ${QOS}, ${GRES}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
sbatch "${SBATCH_ARGS[@]}" "${JOB_SCRIPT}" "$@"
