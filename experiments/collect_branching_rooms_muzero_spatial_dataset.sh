#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

dataset_dir=.local_runs/branching_rooms_muzero_spatial_dataset_seed17
dataset="$dataset_dir/experienced_trajectories.npz"
args=(
  --condition shared_log_auxiliary_spatial_shadow
  --seed 17
  --exp-name "$dataset_dir/collection_after_reset_fix"
  --model-path .local_runs/branching_rooms_shared_log_auxiliary_matched_seed0/ckpt/ckpt_best.pth.tar
  --collection-only
  --episode-max-step 1200
  --num-simulations 25
  --collector-env-num 4
  --n-episode 12
  --batch-size 128
  --spatial-shadow-roots-per-episode 16
  --spatial-shadow-history-window 128
  --spatial-shadow-export-dataset "$dataset"
  --spatial-shadow-export-split-counts 6,2,2,2
  --cuda
  --no-record-collect-video
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
  exec env \
    MPLCONFIGDIR=/tmp/matplotlib-lightzero \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py "${args[@]}"
fi

env \
  MPLCONFIGDIR=/tmp/matplotlib-lightzero \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -u experiments/run_branching_rooms_intrinsic_rnd.py "${args[@]}"

.venv/bin/python - "$dataset" <<'PY'
import sys

import numpy as np

path = sys.argv[1]
with np.load(path, allow_pickle=False) as data:
    observations = data["observations"]
    actions = data["actions"]
    split_ids = data["split_ids"]
    assert observations.shape[:2] == (12, 1201), observations.shape
    assert actions.shape == (12, 1200), actions.shape
    assert np.bincount(split_ids, minlength=4).tolist() == [6, 2, 2, 2]
print(
    f"Validated {path}: observations={observations.shape}, "
    f"actions={actions.shape}, split_counts={[6, 2, 2, 2]}"
)
PY
