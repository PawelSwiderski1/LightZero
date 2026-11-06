#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${REMOTE:-pmswiderski@login.delftblue.tudelft.nl}"
REMOTE_REPO="${REMOTE_REPO:-~/LightZero}"

files=(
    cluster/slurm_a100.sbatch
    cluster/submit_delftblue.sh
    cluster/submit_branching_rooms_recent_transformer_novelty_long_online_delftblue.sh
    experiments/run_branching_rooms_intrinsic_rnd.py
    lzero/entry/train_ensemble_muzero.py
    lzero/policy/ensemble_muzero.py
    lzero/policy/recurrent_sequence_novelty.py
)

cd "${REPO_ROOT}"
for path in "${files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${REPO_ROOT}/${path}" >&2
        exit 2
    fi
done

echo "Synchronizing recent-transformer experiment to ${REMOTE}:${REMOTE_REPO}/"
rsync -azR "${files[@]}" "${REMOTE}:${REMOTE_REPO}/"
