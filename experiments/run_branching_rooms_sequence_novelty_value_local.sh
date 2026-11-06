#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1

MODE=${1:-}
if [[ "$MODE" != "control" && "$MODE" != "value" ]]; then
  echo "Usage: $0 {control|value} [extra experiment arguments...]" >&2
  exit 2
fi
shift

if [[ "$MODE" == "value" ]]; then
  CONDITION=shared_log_auxiliary_sequence_novelty_value
  DEFAULT_EXP_NAME=.local_runs/branching_rooms_sequence_novelty_value_seed0
else
  CONDITION=shared_log_auxiliary_sequence_novelty
  DEFAULT_EXP_NAME=.local_runs/branching_rooms_sequence_novelty_control_sims25_seed0
fi

EXP_NAME=${EXP_NAME:-$DEFAULT_EXP_NAME}
if [[ -e "$EXP_NAME" ]]; then
  echo "Output already exists: $EXP_NAME" >&2
  echo "Set EXP_NAME to a new directory." >&2
  exit 1
fi

exec .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py \
  --condition "$CONDITION" \
  --seed 0 \
  --exp-name "$EXP_NAME" \
  --max-env-step 1024 \
  --max-train-iter 1024 \
  --episode-max-step 128 \
  --num-simulations 25 \
  --intrinsic-tree-leaf-value-weight 1 \
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
