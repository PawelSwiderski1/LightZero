#!/usr/bin/env bash
#SBATCH --job-name=br-spatial-world-model
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

MODE="${1:?Expected mode: source or adapt}"
SOURCE_CHECKPOINT="${2:-}"

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SCRATCH_ROOT="${SCRATCH:-/scratch/${USER}}"
APPTAINER_ROOT="${LIGHTZERO_APPTAINER_ROOT:-${SCRATCH_ROOT}/lightzero_apptainer}"
IMAGE="${LIGHTZERO_APPTAINER_IMAGE:-${APPTAINER_ROOT}/images/pytorch-2.5.1-cuda12.4-cudnn9-devel.sif}"
VENV_DIR="${LIGHTZERO_VENV:-${APPTAINER_ROOT}/venvs/lightzero-pytorch-2.5.1}"
PYTHON="${VENV_DIR}/bin/python"
RUN_ROOT="${LIGHTZERO_RUN_ROOT:-${SCRATCH_ROOT}/lightzero_runs}"
RUN_LABEL="${RUN_LABEL:-branching_rooms_spatial_world_model_${MODE}}"
RUN_DIR="${RUN_ROOT}/${RUN_LABEL}_${SLURM_JOB_ID}"
LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/${RUN_LABEL}_${SLURM_JOB_ID}"
LOCAL_OUTPUT="${LOCAL_ROOT}/output"
LOCAL_DATASET="${LOCAL_ROOT}/experienced_trajectories.npz"

case "${MODE}" in
    source)
        DATASET="${SOURCE_DATASET:-${REPO_ROOT}/.local_runs/branching_rooms_action_novelty_macro_augmented_seed0/experienced_trajectories.npz}"
        ;;
    adapt)
        DATASET="${TARGET_DATASET:-${REPO_ROOT}/.local_runs/branching_rooms_muzero_spatial_dataset_seed17/experienced_trajectories.npz}"
        if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
            echo "adapt mode requires a source checkpoint path" >&2
            exit 2
        fi
        ;;
    *)
        echo "Unknown mode: ${MODE}; expected source or adapt" >&2
        exit 2
        ;;
esac

for path in "${IMAGE}" "${DATASET}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done
if [[ "${MODE}" == "adapt" && ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "Missing source checkpoint: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi
if ! apptainer sif list "${IMAGE}" >/dev/null 2>&1; then
    echo "Unreadable or invalid Apptainer image: ${IMAGE}" >&2
    exit 3
fi

mkdir -p "${LOCAL_OUTPUT}" "${RUN_ROOT}"
cp "${DATASET}" "${LOCAL_DATASET}"

sync_output() {
    mkdir -p "${RUN_DIR}"
    cp -a "${LOCAL_OUTPUT}/." "${RUN_DIR}/"
}
trap sync_output EXIT
trap 'exit 143' TERM INT

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${LOCAL_ROOT}/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${MPLCONFIGDIR}"

RUNTIME=(
    apptainer exec
    --nv
    --bind "${REPO_ROOT}:${REPO_ROOT}"
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
    --bind "${LOCAL_ROOT}:${LOCAL_ROOT}"
    "${IMAGE}"
)

if ! "${RUNTIME[@]}" test -x "${PYTHON}"; then
    echo "Missing container environment: ${PYTHON}" >&2
    echo "Run cluster/setup_hpc_env.sh on the login node first." >&2
    exit 2
fi

COMMON_ARGS=(
    --dataset "${LOCAL_DATASET}"
    --output-dir "${LOCAL_OUTPUT}"
    --horizon 5
    --batch-size 8
    --evaluation-batch-size 4
    --representation-mode spatial_reconstruction
    --direct-reconstruction-weight 2
    --direct-contrastive-weight 1
    --inverse-action-weight 0.1
    --variance-weight 0.1
    --latent-dynamics-weight 0
    --key-dynamics-weight 1
    --rollout-reconstruction-weight 0
    --rollout-contrastive-weight 1
    --selection-metric key_novelty
    --num-workers 4
    --figure-count 4
    --device cuda
)

if [[ "${MODE}" == "source" ]]; then
    COMMAND=(
        "${RUNTIME[@]}"
        "${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_spatial_exploration_world_model.py"
        "${COMMON_ARGS[@]}"
        --seed 0
        --autoencoder-epochs 80
        --dynamics-epochs 600
        --patience 80
        --roots-per-episode 32
        --evaluation-roots-per-episode 0
    )
else
    LOCAL_CHECKPOINT="${LOCAL_ROOT}/source_best.pt"
    cp "${SOURCE_CHECKPOINT}" "${LOCAL_CHECKPOINT}"
    COMMAND=(
        "${RUNTIME[@]}"
        "${PYTHON}" -u "${REPO_ROOT}/experiments/run_branching_rooms_spatial_exploration_world_model.py"
        "${COMMON_ARGS[@]}"
        --seed 17
        --resume-checkpoint "${LOCAL_CHECKPOINT}"
        --autoencoder-epochs 0
        --dynamics-epochs 900
        --patience 80
        --roots-per-episode 64
        --evaluation-roots-per-episode 128
        --learning-rate 1e-4
        --fresh-optimizer-on-resume
        --evaluate-resume-before-training
    )
fi

echo "mode=${MODE}"
echo "dataset=${DATASET}"
echo "run_dir=${RUN_DIR}"
echo "local_output=${LOCAL_OUTPUT}"
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

cd "${REPO_ROOT}"
srun --ntasks=1 "${COMMAND[@]}"

echo "completed_run_dir=${RUN_DIR}"
