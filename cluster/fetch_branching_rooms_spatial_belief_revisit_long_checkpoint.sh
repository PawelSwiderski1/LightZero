#!/usr/bin/env bash

set -euo pipefail

REMOTE="${REMOTE:-pmswiderski@login.delftblue.tudelft.nl}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/pmswiderski/lightzero_runs}"
REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}/hpc_results/branching_rooms_spatial_belief_revisit_long_online_20261015}"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 RUN_NAME CHECKPOINT_ITERATION" >&2
    exit 2
fi

run_name="$1"
checkpoint_iteration="$2"
if [[ ! "${run_name}" =~ ^br_spatial_belief_revisit_long_online_s[0-9]+_[0-9]+$ ]]; then
    echo "Refusing unexpected run directory name: ${run_name}" >&2
    exit 2
fi
if [[ ! "${checkpoint_iteration}" =~ ^[0-9]+$ ]]; then
    echo "Checkpoint iteration must be a non-negative integer." >&2
    exit 2
fi

destination="${DEST_ROOT}/${run_name}/ckpt"
checkpoint_name="iteration_${checkpoint_iteration}.pth.tar"
mkdir -p "${destination}"

rsync -az \
    --partial \
    --timeout=180 \
    --info=progress2 \
    "${REMOTE}:${REMOTE_ROOT}/${run_name}/ckpt/${checkpoint_name}" \
    "${destination}/"

if [[ ! -s "${destination}/${checkpoint_name}" ]]; then
    echo "Checkpoint download is empty: ${destination}/${checkpoint_name}" >&2
    exit 1
fi

echo "Fetched ${destination}/${checkpoint_name}"
