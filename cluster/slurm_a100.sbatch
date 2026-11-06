#!/usr/bin/env bash
#SBATCH --job-name=lightzero-mcar
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
SCRATCH_ROOT="${SCRATCH:-/scratch/${USER}}"
APPTAINER_ROOT="${LIGHTZERO_APPTAINER_ROOT:-${SCRATCH_ROOT}/lightzero_apptainer}"
IMAGE="${LIGHTZERO_APPTAINER_IMAGE:-${APPTAINER_ROOT}/images/pytorch-2.5.1-cuda12.4-cudnn9-devel.sif}"
VENV_DIR="${LIGHTZERO_VENV:-${APPTAINER_ROOT}/venvs/lightzero-pytorch-2.5.1}"
RUN_ROOT="${LIGHTZERO_RUN_ROOT:-${SCRATCH_ROOT}/lightzero_runs}"
EXPERIMENT="${1:-experiments/run_mountain_car_state_density_intrinsic_diagnostic.py}"
if [[ $# -gt 0 ]]; then
    shift
fi

SEED="${SEED:-0}"
RUN_LABEL="${RUN_LABEL:-$(basename "${EXPERIMENT}" .py)_seed${SEED}}"
RUN_DIR="${RUN_ROOT}/${RUN_LABEL}_${SLURM_JOB_ID}"
RUN_LINK="${REPO_ROOT}/.hpc_runs"
EXP_NAME=".hpc_runs/${RUN_LABEL}_${SLURM_JOB_ID}"
PYTHON="${VENV_DIR}/bin/python"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "Apptainer is not available on PATH." >&2
    exit 2
fi
if [[ ! -f "${IMAGE}" ]]; then
    echo "Missing Apptainer image: ${IMAGE}" >&2
    echo "Run cluster/setup_hpc_env.sh on the login node first." >&2
    exit 2
fi
if ! apptainer sif list "${IMAGE}" >/dev/null 2>&1; then
    echo "Unreadable or invalid Apptainer image: ${IMAGE}" >&2
    echo "The image exists, so rerunning setup in place will not replace it." >&2
    echo "Check for a transient filesystem error or select a verified recovery root." >&2
    exit 3
fi

RUNTIME=(
    apptainer exec
    --nv
    --bind "${REPO_ROOT}:${REPO_ROOT}"
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
    "${IMAGE}"
)

if ! "${RUNTIME[@]}" test -x "${PYTHON}"; then
    echo "Missing container environment: ${PYTHON}" >&2
    echo "Run cluster/setup_hpc_env.sh on the login node first." >&2
    exit 2
fi
if [[ ! -f "${REPO_ROOT}/${EXPERIMENT}" ]]; then
    echo "Experiment does not exist: ${REPO_ROOT}/${EXPERIMENT}" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}"
if [[ -e "${RUN_LINK}" && ! -L "${RUN_LINK}" ]]; then
    echo "Cannot create scratch link because this path is not a symlink: ${RUN_LINK}" >&2
    exit 2
fi
ln -sfn "${RUN_ROOT}" "${RUN_LINK}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/lightzero-matplotlib-${SLURM_JOB_ID}"
export TORCH_HOME="${TORCH_HOME:-${RUN_ROOT}/.torch}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_ROOT}/cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}"

cd "${REPO_ROOT}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "experiment=${EXPERIMENT}"
echo "run_dir_prefix=${RUN_DIR}"
echo "experiment_name=${EXP_NAME}"
echo "apptainer_image=${IMAGE}"
nvidia-smi
"${RUNTIME[@]}" "${PYTHON}" -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"

COMMAND=(
    "${RUNTIME[@]}"
    "${PYTHON}" -u "${REPO_ROOT}/${EXPERIMENT}"
    --exp-name "${EXP_NAME}"
    --seed "${SEED}"
    --cuda
    --no-record-collect-video
)
COMMAND+=("$@")

printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

srun --ntasks=1 "${COMMAND[@]}"

echo "completed_run_dir=${RUN_DIR}"
