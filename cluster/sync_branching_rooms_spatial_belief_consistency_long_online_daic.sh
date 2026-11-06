#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${REMOTE:-pmswiderski@login.daic.tudelft.nl}"
REMOTE_REPO="${REMOTE_REPO:-~/LightZero}"

files=(
    cluster/requirements-branching-rooms-daic.txt
    cluster/setup_daic_native_env.sh
    cluster/slurm_daic_a40.sbatch
    cluster/submit_daic.sh
    cluster/submit_branching_rooms_spatial_belief_consistency_long_online_daic.sh
    experiments/run_branching_rooms_intrinsic_rnd.py
    experiments/run_branching_rooms_spatial_exploration_world_model.py
    lzero/entry/spatial_exploration_shadow.py
    lzero/entry/train_ensemble_muzero.py
    lzero/mcts/buffer/game_buffer_unizero.py
    lzero/model/common.py
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

echo "Synchronizing DAIC belief-consistency experiment to ${REMOTE}:${REMOTE_REPO}/"
rsync -azR "${files[@]}" "${REMOTE}:${REMOTE_REPO}/"
