#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1
EXP_NAME=${EXP_NAME:-.local_runs/branching_rooms_shared_log_auxiliary_sequence_novelty_seed0}

exec .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py \
  --condition shared_log_auxiliary_sequence_novelty \
  --seed 0 \
  --exp-name "$EXP_NAME" \
  --max-env-step 1024 \
  --max-train-iter 1024 \
  --episode-max-step 128 \
  --num-simulations 5 \
  --collector-env-num 1 \
  --n-episode 1 \
  --batch-size 16 \
  --replay-ratio 0.5 \
  --rnd-fresh-update-batch-size 16 \
  --episodic-sequence-updates-per-collect 64 \
  --episodic-sequence-train-batch-size 1 \
  --episodic-sequence-roots-per-update 32 \
  --cuda \
  --no-record-collect-video \
  "$@"
