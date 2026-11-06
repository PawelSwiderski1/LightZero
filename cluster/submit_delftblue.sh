#!/usr/bin/env bash

set -euo pipefail

PARTITION="${PARTITION:-gpu-a100}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_CPU="${MEM_PER_CPU:-2GB}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="${REPO_ROOT}/cluster/slurm_a100.sbatch"

SBATCH_ARGS=(
    --partition="${PARTITION}"
    --gpus-per-task="${GPUS_PER_TASK}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem-per-cpu="${MEM_PER_CPU}"
    --time="${TIME_LIMIT}"
)

if [[ -n "${RUN_LABEL:-}" ]]; then
    SBATCH_ARGS+=(--job-name="${RUN_LABEL}")
fi

if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    SBATCH_ARGS+=(--account="${SLURM_ACCOUNT}")
fi

echo "Submitting to ${PARTITION}: ${GPUS_PER_TASK} GPU, ${CPUS_PER_TASK} CPUs, ${MEM_PER_CPU}/CPU, ${TIME_LIMIT}"
sbatch "${SBATCH_ARGS[@]}" "${JOB_SCRIPT}" "$@"
