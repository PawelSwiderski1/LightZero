#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1

OUTPUT_DIR=${OUTPUT_DIR:-.local_runs/branching_rooms_action_novelty_q_epoch90_behavior_seed0}

.venv/bin/python -u experiments/run_branching_rooms_action_novelty_q.py \
  --offline-run .local_runs/branching_rooms_action_novelty_seed0 \
  --q-checkpoint \
    .local_runs/branching_rooms_action_novelty_q_convergence_seed0/epoch_090.pt \
  --output-dir "$OUTPUT_DIR" \
  --device cpu \
  "$@"

.venv/bin/python -u \
  experiments/diagnose_branching_rooms_action_novelty_q_decisions.py \
  --run-dir "$OUTPUT_DIR" \
  --device cpu
