#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# Extrinsic evaluation is always zero in this environment, so ckpt_best can
# remain the untrained initial checkpoint and omit the separately trained RND state.
python_bin=${PYTHON_BIN:-.venv/bin/python}
muzero_checkpoint=${MUZERO_CHECKPOINT:-.local_runs/branching_rooms_shared_log_auxiliary_matched_seed0/ckpt/iteration_512.pth.tar}
spatial_checkpoint=${SPATIAL_CHECKPOINT:-hpc_results/branching_rooms_spatial_world_model_long_20261011/br_spatial_key_long_source_s0_951676/best.pt}
episode_steps=${EPISODE_STEPS:-200}
num_simulations=${NUM_SIMULATIONS:-25}
leaf_value_weight=${LEAF_VALUE_WEIGHT:-0.01}
beta=${SPATIAL_BETA:-0.05}
reward_mode=${SPATIAL_REWARD_MODE:-gate}
# Keep the current experiment focused on episodic revisit correction. Direct
# evaluation of predicted observations with RND is a separate, opt-in ablation.
rnd_estimator=${SPATIAL_RND_ESTIMATOR:-head}
rnd_hybrid_weight=${SPATIAL_RND_HYBRID_WEIGHT:-0.5}
minimum_reward_factor=${SPATIAL_MINIMUM_REWARD_FACTOR:-0.05}
seed=${SEED:-0}
run_tag=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
output_root=${OUTPUT_ROOT:-.local_runs/branching_rooms_spatial_tree_comparison_${run_tag}}

for path in "$python_bin" "$muzero_checkpoint" "$spatial_checkpoint"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 2
    fi
done

device_flag=--cuda
if [[ "${DEVICE:-cuda}" == "cpu" ]]; then
    device_flag=--no-cuda
fi

common=(
    --seed "$seed"
    --model-path "$muzero_checkpoint"
    --spatial-exploration-checkpoint "$spatial_checkpoint"
    --collection-only
    --episode-max-step "$episode_steps"
    --num-simulations "$num_simulations"
    --intrinsic-tree-leaf-value-weight "$leaf_value_weight"
    --collector-env-num 1
    --n-episode 1
    --batch-size 16
    --no-record-collect-video
    "$device_flag"
)

run_condition() {
    local name=$1
    local condition=$2
    shift 2
    local output_dir="${output_root}/${name}"
    mkdir -p "$output_dir"
    echo "===== ${name} ====="
    env MPLCONFIGDIR=/tmp/matplotlib-lightzero PYTHONUNBUFFERED=1 \
        "$python_bin" -u experiments/run_branching_rooms_intrinsic_rnd.py \
        --condition "$condition" \
        --exp-name "$output_dir" \
        "${common[@]}" \
        "$@" \
        2>&1 | tee "$output_dir/run.log"
}

# All conditions load the same MuZero and spatial checkpoints. The control only
# evaluates the spatial model in shadow mode; it does not alter MCTS rewards.
run_condition baseline shared_log_auxiliary_spatial_shadow
run_condition depth1 shared_log_auxiliary_spatial_tree \
    --spatial-tree-max-depth 1 \
    --spatial-tree-reward-mode "$reward_mode" \
    --spatial-tree-rnd-estimator "$rnd_estimator" \
    --spatial-tree-rnd-hybrid-weight "$rnd_hybrid_weight" \
    --spatial-tree-beta "$beta" \
    --spatial-tree-minimum-reward-factor "$minimum_reward_factor"
run_condition depth5 shared_log_auxiliary_spatial_tree \
    --spatial-tree-max-depth 5 \
    --spatial-tree-reward-mode "$reward_mode" \
    --spatial-tree-rnd-estimator "$rnd_estimator" \
    --spatial-tree-rnd-hybrid-weight "$rnd_hybrid_weight" \
    --spatial-tree-beta "$beta" \
    --spatial-tree-minimum-reward-factor "$minimum_reward_factor"

echo "Results: $output_root"
