#!/usr/bin/env bash

set -euo pipefail

REMOTE="${REMOTE:-pmswiderski@login.delftblue.tudelft.nl}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/pmswiderski/lightzero_runs}"
REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST_ROOT="${DEST_ROOT:-${REPO_ROOT}/hpc_results/branching_rooms_recent_transformer_novelty_long_online_20261020}"

if [[ $# -gt 0 ]]; then
    run_names=("$@")
else
    mapfile -t run_names < <(
        ssh "${REMOTE}" \
            "find '${REMOTE_ROOT}' -mindepth 1 -maxdepth 1 -type d -name 'br_recent_transformer_novelty_long_online_s*_*' -printf '%f\\n' | sort"
    )
fi

if [[ ${#run_names[@]} -eq 0 ]]; then
    echo "No long recent-transformer runs found under ${REMOTE_ROOT}." >&2
    exit 1
fi

mkdir -p "${DEST_ROOT}"

for run_name in "${run_names[@]}"; do
    if [[ ! "${run_name}" =~ ^br_recent_transformer_novelty_long_online_s[0-9]+_[0-9]+$ ]]; then
        echo "Refusing unexpected run directory name: ${run_name}" >&2
        exit 2
    fi

    destination="${DEST_ROOT}/${run_name}"
    mkdir -p "${destination}"
    echo "Fetching ${run_name}"
    rsync -az \
        --partial \
        --timeout=180 \
        --info=progress2 \
        --exclude='/ckpt/' \
        "${REMOTE}:${REMOTE_ROOT}/${run_name}/" \
        "${destination}/"

    for required in \
        diagnostics/environment_coverage.csv \
        diagnostics/episodic_sequence_training.csv \
        diagnostics/episodic_sequence_root_action_summary.csv; do
        if [[ ! -s "${destination}/${required}" ]]; then
            echo "Missing required result: ${destination}/${required}" >&2
            exit 1
        fi
    done
done

echo "Fetched ${#run_names[@]} run(s) to ${DEST_ROOT}"
