#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-0}
EXP_NAME=${EXP_NAME:-.local_runs/branching_rooms_reconstructive_memory_seed${SEED}}
MAX_ENV_STEP=${MAX_ENV_STEP:-1024}
MAX_TRAIN_ITER=${MAX_TRAIN_ITER:-1024}
EPISODE_MAX_STEP=${EPISODE_MAX_STEP:-128}
NUM_SIMULATIONS=${NUM_SIMULATIONS:-5}
SIDECAR_UPDATES=${SIDECAR_UPDATES:-256}

exec env \
  MPLCONFIGDIR=/tmp/matplotlib-lightzero \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py \
  --condition shared_log_auxiliary_reconstructive_memory \
  --seed "$SEED" \
  --exp-name "$EXP_NAME" \
  --max-env-step "$MAX_ENV_STEP" \
  --max-train-iter "$MAX_TRAIN_ITER" \
  --episode-max-step "$EPISODE_MAX_STEP" \
  --num-simulations "$NUM_SIMULATIONS" \
  --collector-env-num 1 \
  --n-episode 1 \
  --batch-size 16 \
  --replay-ratio 0.5 \
  --rnd-fresh-update-batch-size 16 \
  --episodic-sequence-updates-per-collect "$SIDECAR_UPDATES" \
  --episodic-sequence-train-batch-size 1 \
  --episodic-sequence-roots-per-update 32 \
  --cuda \
  --no-record-collect-video \
  "$@"
