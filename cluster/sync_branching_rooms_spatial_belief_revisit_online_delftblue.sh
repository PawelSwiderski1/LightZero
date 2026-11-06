#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${REMOTE:-pmswiderski@login.delftblue.tudelft.nl}"
REMOTE_REPO="${REMOTE_REPO:-~/LightZero}"

files=(
    cluster/slurm_a100.sbatch
    cluster/submit_delftblue.sh
    cluster/submit_branching_rooms_spatial_belief_revisit_online_delftblue.sh
    cluster/fetch_branching_rooms_spatial_belief_revisit_online.sh
    experiments/run_branching_rooms_intrinsic_rnd.py
    experiments/run_branching_rooms_spatial_exploration_world_model.py
    lzero/entry/spatial_exploration_shadow.py
    lzero/entry/train_ensemble_muzero.py
    lzero/policy/ensemble_muzero.py
    lzero/policy/spatial_exploration_tree.py
)

cd "${REPO_ROOT}"
for path in "${files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${REPO_ROOT}/${path}" >&2
        exit 2
    fi
done

echo "Synchronizing pairwise belief-revisit online experiment to ${REMOTE}:${REMOTE_REPO}/"
rsync -azR "${files[@]}" "${REMOTE}:${REMOTE_REPO}/"
