#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

dataset=${DATASET:-.local_runs/branching_rooms_muzero_spatial_dataset_seed17/experienced_trajectories.npz}
checkpoint=${CHECKPOINT:-.local_runs/branching_rooms_spatial_exploration_world_model_macro_training_seed0/best.pt}
output=${OUTPUT_DIR:-.local_runs/branching_rooms_spatial_world_model_muzero_key_only_seed17}
device=${DEVICE:-cuda}

for path in "$dataset" "$checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

command=(
  .venv/bin/python -u experiments/run_branching_rooms_spatial_exploration_world_model.py
  --dataset "$dataset"
  --output-dir "$output"
  --resume-checkpoint "$checkpoint"
  --seed 17
  --horizon 5
  --dynamics-epochs 220
  --patience 20
  --batch-size 2
  --evaluation-batch-size 1
  --roots-per-episode 64
  --evaluation-roots-per-episode 128
  --learning-rate 1e-4
  --latent-dynamics-weight 0
  --key-dynamics-weight 1
  --rollout-reconstruction-weight 0
  --rollout-contrastive-weight 1
  --selection-metric key_novelty
  --fresh-optimizer-on-resume
  --evaluate-resume-before-training
  --figure-count 4
  --device "$device"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'MPLCONFIGDIR=/tmp/matplotlib-lightzero PYTHONUNBUFFERED=1'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec env \
  MPLCONFIGDIR=/tmp/matplotlib-lightzero \
  PYTHONUNBUFFERED=1 \
  "${command[@]}"
