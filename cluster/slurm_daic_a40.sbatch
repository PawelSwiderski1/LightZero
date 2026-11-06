#!/usr/bin/env bash
#SBATCH --job-name=lightzero-mcar
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=4GB
#SBATCH --mail-type=END
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
DAIC_STORAGE_ROOT="${DAIC_STORAGE_ROOT:-${HOME}}"
RUNTIME_MODE="${LIGHTZERO_RUNTIME:-apptainer}"
APPTAINER_ROOT="${LIGHTZERO_APPTAINER_ROOT:-${DAIC_STORAGE_ROOT}/lightzero_apptainer}"
IMAGE_SOURCE="${LIGHTZERO_APPTAINER_SOURCE:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
IMAGE="${LIGHTZERO_APPTAINER_IMAGE:-${APPTAINER_ROOT}/images/pytorch-2.5.1-cuda12.4-cudnn9-devel.sif}"
RUN_ROOT="${LIGHTZERO_RUN_ROOT:-${DAIC_STORAGE_ROOT}/lightzero_runs}"
EXPERIMENT="${1:-experiments/run_mountain_car_encoder_random_prior.py}"
if [[ $# -gt 0 ]]; then
    shift
fi

SEED="${SEED:-0}"
RUN_LABEL="${RUN_LABEL:-$(basename "${EXPERIMENT}" .py)_seed${SEED}}"
RUN_DIR="${RUN_ROOT}/${RUN_LABEL}_${SLURM_JOB_ID}"
RUN_LINK="${REPO_ROOT}/.daic_runs"
EXP_NAME=".daic_runs/${RUN_LABEL}_${SLURM_JOB_ID}"
JOB_TMP="${TMPDIR:-/tmp/${USER}}/lightzero-${SLURM_JOB_ID}"
if [[ "${RUNTIME_MODE}" == "native" ]]; then
    VENV_DIR="${LIGHTZERO_VENV:-${HOME}/.venvs/lightzero-pytorch-2.5.1-cu124}"
elif [[ "${RUNTIME_MODE}" == "ephemeral_apptainer" ]]; then
    VENV_DIR="${JOB_TMP}/venv"
else
    VENV_DIR="${LIGHTZERO_VENV:-${APPTAINER_ROOT}/venvs/lightzero-pytorch-2.5.1}"
fi
PYTHON="${VENV_DIR}/bin/python"

if [[ ! -f "${REPO_ROOT}/${EXPERIMENT}" ]]; then
    echo "Experiment does not exist: ${REPO_ROOT}/${EXPERIMENT}" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}" "${JOB_TMP}" "${REPO_ROOT}/slurm_logs"
cleanup_job_tmp() {
    rm -rf "${JOB_TMP}"
}
trap cleanup_job_tmp EXIT
if [[ -e "${RUN_LINK}" && ! -L "${RUN_LINK}" ]]; then
    echo "Cannot create DAIC run link because this path is not a symlink: ${RUN_LINK}" >&2
    exit 2
fi
ln -sfn "${RUN_ROOT}" "${RUN_LINK}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${JOB_TMP}/matplotlib"
if [[ "${RUNTIME_MODE}" == "ephemeral_apptainer" ]]; then
    export TORCH_HOME="${JOB_TMP}/torch"
else
    export TORCH_HOME="${TORCH_HOME:-${DAIC_STORAGE_ROOT}/.cache/torch}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}"
cd "${REPO_ROOT}"

RUNTIME=()
if [[ "${RUNTIME_MODE}" == "native" ]]; then
    module use /opt/insy/modulefiles
    module load miniconda/3.10
    if [[ ! -x "${PYTHON}" ]]; then
        echo "Missing native environment: ${PYTHON}" >&2
        echo "Run cluster/setup_daic_native_env.sh on the login node first." >&2
        exit 2
    fi
elif [[ "${RUNTIME_MODE}" == "apptainer" || "${RUNTIME_MODE}" == "ephemeral_apptainer" ]]; then
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "Apptainer is not available on PATH." >&2
        exit 2
    fi
    if [[ "${RUNTIME_MODE}" == "apptainer" && ! -f "${IMAGE}" ]]; then
        echo "Missing Apptainer image: ${IMAGE}" >&2
        echo "Run cluster/setup_daic_env.sh on the login node first." >&2
        exit 2
    fi
    if [[ "${RUNTIME_MODE}" == "ephemeral_apptainer" ]]; then
        IMAGE="${IMAGE_SOURCE}"
        export APPTAINER_CACHEDIR="${JOB_TMP}/apptainer-cache"
        export APPTAINER_TMPDIR="${JOB_TMP}/apptainer"
    else
        export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_ROOT}/cache}"
        export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${JOB_TMP}/apptainer}"
    fi
    mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
    RUNTIME=(
        apptainer exec
        --nv
        --bind "${REPO_ROOT}:${REPO_ROOT}"
        --bind "${DAIC_STORAGE_ROOT}:${DAIC_STORAGE_ROOT}"
        --bind "${JOB_TMP}:${JOB_TMP}"
        "${IMAGE}"
    )
    if [[ "${RUNTIME_MODE}" == "ephemeral_apptainer" ]]; then
        REQUIREMENTS_FILE="${LIGHTZERO_REQUIREMENTS_FILE:-${REPO_ROOT}/cluster/requirements-branching-rooms-daic.txt}"
        if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
            echo "Requirements file does not exist: ${REQUIREMENTS_FILE}" >&2
            exit 2
        fi
        export PIP_NO_CACHE_DIR=1
        echo "Preparing ephemeral environment from ${IMAGE_SOURCE}"
        "${RUNTIME[@]}" python -m venv --system-site-packages "${VENV_DIR}"
        "${RUNTIME[@]}" "${PYTHON}" -m pip install \
            pip==23.3.2 setuptools==66.1.1 wheel==0.42.0
        "${RUNTIME[@]}" "${PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"
        "${RUNTIME[@]}" "${PYTHON}" "${REPO_ROOT}/setup.py" build_ext --inplace
    elif ! "${RUNTIME[@]}" test -x "${PYTHON}"; then
        echo "Missing container environment: ${PYTHON}" >&2
        echo "Run cluster/setup_daic_env.sh on the login node first." >&2
        exit 2
    fi
else
    echo "Unsupported LIGHTZERO_RUNTIME: ${RUNTIME_MODE}" >&2
    exit 2
fi

cd "${REPO_ROOT}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "experiment=${EXPERIMENT}"
echo "run_dir=${RUN_DIR}"
echo "experiment_name=${EXP_NAME}"
echo "runtime_mode=${RUNTIME_MODE}"
if [[ "${RUNTIME_MODE}" == "apptainer" || "${RUNTIME_MODE}" == "ephemeral_apptainer" ]]; then
    echo "apptainer_image=${IMAGE}"
fi
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
