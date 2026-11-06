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
RUN_NAME="${1:-br_spatial_belief_revisit_priority_long_online_s0_1021540}"
RUN_ROOT="${REPO_ROOT}/hpc_results/branching_rooms_spatial_belief_revisit_priority_long_online_20261018/${RUN_NAME}"
EXTRACTOR="experiments/extract_spatial_exploration_online_checkpoint.py"
INCLUDE_REPLAY="${INCLUDE_REPLAY:-0}"

if [[ "${INCLUDE_REPLAY}" == "1" ]]; then
    output_name="spatial_exploration_tree_online_with_replay.pt"
    extractor_option="--include-replay"
else
    output_name="spatial_exploration_tree_online.pt"
    extractor_option=""
fi

if [[ ! "${RUN_NAME}" =~ ^br_spatial_belief_revisit_priority_long_online_s[0-9]+_[0-9]+$ ]]; then
    echo "Refusing unexpected run directory name: ${RUN_NAME}" >&2
    exit 2
fi

remote_run="${REMOTE_RUN_ROOT}/${RUN_NAME}"
remote_output="${remote_run}/${output_name}"
local_output="${RUN_ROOT}/${output_name}"
control_path="${SSH_CONTROL_PATH:-/tmp/lightzero-fetch-${USER}-$$.sock}"

cleanup() {
    ssh -S "${control_path}" -O exit "${REMOTE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${RUN_ROOT}"
echo "Opening a reusable DelftBlue SSH connection"
ssh -M -S "${control_path}" -fN -o ControlPersist=300 "${REMOTE}"

rsync -az -e "ssh -S ${control_path}" \
    "${REPO_ROOT}/${EXTRACTOR}" \
    "${REMOTE}:${REMOTE_REPO}/experiments/"

echo "Extracting the latest spatial model from ${RUN_NAME} on DelftBlue (include_replay=${INCLUDE_REPLAY})"
ssh -S "${control_path}" \
    "${REMOTE}" \
    "apptainer exec --bind '${REMOTE_REPO}:${REMOTE_REPO}' --bind '${REMOTE_SCRATCH}:${REMOTE_SCRATCH}' '${REMOTE_IMAGE}' '${REMOTE_PYTHON}' -u '${REMOTE_REPO}/${EXTRACTOR}' --checkpoint-dir '${remote_run}/ckpt' --output '${remote_output}' ${extractor_option}"

rsync -az -e "ssh -S ${control_path}" --partial --info=progress2 \
    "${REMOTE}:${remote_output}" \
    "${local_output}"

if [[ ! -s "${local_output}" ]]; then
    echo "Extracted model download is empty: ${local_output}" >&2
    exit 1
fi

echo "Fetched ${local_output}"
