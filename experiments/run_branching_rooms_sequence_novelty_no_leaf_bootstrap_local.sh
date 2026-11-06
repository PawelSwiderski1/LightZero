#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1

CHECKPOINT=${CHECKPOINT:-.local_runs/branching_rooms_sequence_novelty_candidate_diagnostics_seed0/ckpt/iteration_512.pth.tar}
EXP_NAME=${EXP_NAME:-.local_runs/branching_rooms_sequence_novelty_checkpoint_sims25_no_leaf_bootstrap_seed0}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ -e "$EXP_NAME" ]]; then
  echo "Output already exists: $EXP_NAME" >&2
  echo "Remove it or set EXP_NAME to a new directory." >&2
  exit 1
fi

exec .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py \
  --condition shared_log_auxiliary_sequence_novelty \
  --seed 0 \
  --exp-name "$EXP_NAME" \
  --model-path "$CHECKPOINT" \
  --collection-only \
  --max-env-step 128 \
  --max-train-iter 1 \
  --episode-max-step 128 \
  --num-simulations 25 \
  --intrinsic-tree-leaf-value-weight 0 \
  --collector-env-num 1 \
  --n-episode 1 \
  --batch-size 16 \
  --replay-ratio 0.0 \
  --rnd-fresh-update-batch-size 4096 \
  --episodic-sequence-updates-per-collect 1 \
  --episodic-sequence-train-batch-size 1 \
  --episodic-sequence-roots-per-update 1 \
  --cuda \
  --no-record-collect-video
