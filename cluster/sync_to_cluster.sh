#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 USER@HOST [REMOTE_PATH]" >&2
    echo "REMOTE_PATH defaults to LightZero in the remote home directory." >&2
    exit 2
fi

REMOTE="$1"
REMOTE_PATH="${2:-LightZero}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_ARGS=(
    -az
    --info=progress2
    --exclude '/.git/'
    --exclude '/.venv/'
    --exclude '/.venv-*/'
    --exclude '/.agents/'
    --exclude '/.codex/'
    --exclude '/.idea/'
    --exclude '/.hpc_runs'
    --exclude '/.daic_runs'
    --exclude '/.local_runs/'
    --exclude '/.pytest_cache/'
    --exclude '/.mypy_cache/'
    --exclude '/.github/'
    --exclude '/LightZero.egg-info/'
    --exclude '/assets/'
    --exclude '/docs/'
    --exclude '/tmp/'
    --exclude '/hpc_results/'
    --exclude 'data_*/'
    --exclude '/lz_result/'
    --exclude '/video/'
    --exclude '/build/'
    --exclude '*.so'
    --exclude '*.pyc'
    --exclude '__pycache__/'
    --exclude '*.pth.tar'
    --exclude '*.mp4'
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    RSYNC_ARGS+=(--dry-run --stats)
    rsync "${RSYNC_ARGS[@]}" "${REPO_ROOT}/" "${REMOTE}:${REMOTE_PATH}/"
    exit 0
fi

ssh "${REMOTE}" "mkdir -p '${REMOTE_PATH}'"
rsync "${RSYNC_ARGS[@]}" "${REPO_ROOT}/" "${REMOTE}:${REMOTE_PATH}/"

echo "synced_to=${REMOTE}:${REMOTE_PATH}"
