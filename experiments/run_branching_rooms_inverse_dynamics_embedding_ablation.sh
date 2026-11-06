#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python=${PYTHON:-.venv/bin/python}
device=${DEVICE:-cuda}
macro_dataset=${MACRO_DATASET:-.local_runs/branching_rooms_action_novelty_macro_augmented_seed0/experienced_trajectories.npz}
muzero_dataset=${MUZERO_DATASET:-.local_runs/branching_rooms_muzero_spatial_dataset_seed17/experienced_trajectories.npz}
source_output=${SOURCE_OUTPUT_DIR:-.local_runs/branching_rooms_inverse_embedding_world_model_macro_seed0}
adapted_output=${ADAPTED_OUTPUT_DIR:-.local_runs/branching_rooms_inverse_embedding_world_model_muzero_seed17}

for path in "$macro_dataset" "$muzero_dataset"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required dataset: $path" >&2
    exit 1
  fi
done

representation_args=(
  --key-dim 64
  --representation-mode inverse_dynamics
  --direct-reconstruction-weight 0
  --direct-contrastive-weight 0
  --inverse-action-weight 1
  --variance-weight 0.1
  --latent-dynamics-weight 0
  --key-dynamics-weight 1
  --rollout-reconstruction-weight 0
  --rollout-contrastive-weight 0
  --selection-metric key_novelty
  --figure-count 0
)

source_command=(
  "$python" -u experiments/run_branching_rooms_spatial_exploration_world_model.py
  --dataset "$macro_dataset"
  --output-dir "$source_output"
  --seed 0
  --horizon 5
  --autoencoder-epochs 20
  --dynamics-epochs 160
  --patience 20
  --batch-size 8
  --evaluation-batch-size 4
  --roots-per-episode 32
  "${representation_args[@]}"
  --device "$device"
)

adapted_command=(
  "$python" -u experiments/run_branching_rooms_spatial_exploration_world_model.py
  --dataset "$muzero_dataset"
  --output-dir "$adapted_output"
  --resume-checkpoint "$source_output/best.pt"
  --seed 17
  --horizon 5
  --dynamics-epochs 220
  --patience 20
  --batch-size 2
  --evaluation-batch-size 1
  --roots-per-episode 64
  --evaluation-roots-per-episode 128
  --learning-rate 1e-4
  "${representation_args[@]}"
  --fresh-optimizer-on-resume
  --evaluate-resume-before-training
  --device "$device"
)

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-lightzero}
export PYTHONUNBUFFERED=1

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'MPLCONFIGDIR=%q PYTHONUNBUFFERED=1' "$MPLCONFIGDIR"
  printf ' %q' "${source_command[@]}"
  printf '\n'
  printf 'MPLCONFIGDIR=%q PYTHONUNBUFFERED=1' "$MPLCONFIGDIR"
  printf ' %q' "${adapted_command[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "$source_output/best.pt" ]]; then
  mkdir -p "$source_output"
  "${source_command[@]}" 2>&1 | tee -a "${source_output}.log"
else
  echo "Using existing inverse-dynamics source checkpoint: $source_output/best.pt"
fi

mkdir -p "$adapted_output"
"${adapted_command[@]}" 2>&1 | tee -a "${adapted_output}.log"
