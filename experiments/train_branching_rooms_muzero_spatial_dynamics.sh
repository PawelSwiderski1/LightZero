#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

dataset=.local_runs/branching_rooms_muzero_spatial_dataset_seed17/experienced_trajectories.npz
checkpoint=.local_runs/branching_rooms_spatial_exploration_world_model_macro_training_seed0/best.pt
output=.local_runs/branching_rooms_spatial_world_model_muzero_adapted_seed17

if [[ ! -f "$dataset" ]]; then
  echo "Missing dataset: $dataset" >&2
  echo "Run experiments/collect_branching_rooms_muzero_spatial_dataset.sh first." >&2
  exit 1
fi

exec env \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -u experiments/run_branching_rooms_spatial_exploration_world_model.py \
  --dataset "$dataset" \
  --output-dir "$output" \
  --resume-checkpoint "$checkpoint" \
  --seed 17 \
  --horizon 5 \
  --dynamics-epochs 220 \
  --patience 20 \
  --batch-size 2 \
  --evaluation-batch-size 1 \
  --roots-per-episode 64 \
  --evaluation-roots-per-episode 128 \
  --learning-rate 1e-4 \
  --fresh-optimizer-on-resume \
  --reset-selection-on-resume \
  --evaluate-resume-before-training \
  --figure-count 4 \
  --device cuda
