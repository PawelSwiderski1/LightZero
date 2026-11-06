#!/usr/bin/env bash

set -euo pipefail

REMOTE="${REMOTE:-pmswiderski@login.delftblue.tudelft.nl}"
REMOTE_HOME="${REMOTE_HOME:-/home/pmswiderski}"
REMOTE_REPO="${REMOTE_REPO:-${REMOTE_HOME}/LightZero}"
REMOTE_SCRATCH="${REMOTE_SCRATCH:-/scratch/pmswiderski}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-${REMOTE_SCRATCH}/lightzero_runs}"
REMOTE_APPTAINER_ROOT="${REMOTE_APPTAINER_ROOT:-${REMOTE_SCRATCH}/lightzero_apptainer}"
REMOTE_IMAGE="${REMOTE_IMAGE:-${REMOTE_APPTAINER_ROOT}/images/pytorch-2.5.1-cuda12.4-cudnn9-devel.sif}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_APPTAINER_ROOT}/venvs/lightzero-pytorch-2.5.1/bin/python}"
REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}/hpc_results/branching_rooms_spatial_revisit_controlled_comparison_20261015/checkpoints}"
EXTRACTOR="experiments/extract_spatial_exploration_online_checkpoint.py"

runs=(
    br_spatial_belief_revisit_online_s0_990318
    br_spatial_belief_keycons_online_s0_995899
)

mkdir -p "${DEST_ROOT}"
rsync -az "${REPO_ROOT}/${EXTRACTOR}" "${REMOTE}:${REMOTE_REPO}/experiments/"

for run_name in "${runs[@]}"; do
    remote_run="${REMOTE_RUN_ROOT}/${run_name}"
    remote_output="${remote_run}/spatial_exploration_tree_online.pt"
    echo "Extracting ${run_name} on DelftBlue"
    ssh "${REMOTE}" \
        "apptainer exec --bind '${REMOTE_REPO}:${REMOTE_REPO}' --bind '${REMOTE_SCRATCH}:${REMOTE_SCRATCH}' '${REMOTE_IMAGE}' '${REMOTE_PYTHON}' -u '${REMOTE_REPO}/${EXTRACTOR}' --checkpoint-dir '${remote_run}/ckpt' --output '${remote_output}'"
    rsync -az --partial --info=progress2 \
        "${REMOTE}:${remote_output}" \
        "${DEST_ROOT}/${run_name}.pt"
done

echo "Fetched ${#runs[@]} extracted checkpoints to ${DEST_ROOT}"
