#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

args=(
  --condition shared_log_auxiliary_spatial_shadow
  --seed 0
  --exp-name .local_runs/branching_rooms_spatial_shadow_checkpoint_eval_seed0_bounded128
  --model-path .local_runs/branching_rooms_shared_log_auxiliary_matched_seed0/ckpt/ckpt_best.pth.tar
  --collection-only
  --episode-max-step 1200
  --num-simulations 25
  --collector-env-num 4
  --n-episode 4
  --batch-size 128
  --spatial-shadow-roots-per-episode 128
  --spatial-shadow-history-window 128
  --cuda
  --no-record-collect-video
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

exec env \
  MPLCONFIGDIR=/tmp/matplotlib-lightzero \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py "${args[@]}"
