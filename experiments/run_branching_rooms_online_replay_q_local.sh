#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1

exec .venv/bin/python -u experiments/run_branching_rooms_online_replay_q.py \
  --output-dir .local_runs/branching_rooms_online_replay_q_seed0 \
  --device cpu \
  "$@"
