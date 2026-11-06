#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DAIC_STORAGE_ROOT="${DAIC_STORAGE_ROOT:-${HOME}}"
APPTAINER_ROOT="${LIGHTZERO_APPTAINER_ROOT:-${DAIC_STORAGE_ROOT}/lightzero_apptainer}"
IMAGE_SOURCE="${LIGHTZERO_APPTAINER_SOURCE:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
IMAGE="${LIGHTZERO_APPTAINER_IMAGE:-${APPTAINER_ROOT}/images/pytorch-2.5.1-cuda12.4-cudnn9-devel.sif}"
VENV_DIR="${LIGHTZERO_VENV:-${APPTAINER_ROOT}/venvs/lightzero-pytorch-2.5.1}"
REQUIREMENTS_FILE="${LIGHTZERO_REQUIREMENTS_FILE:-${REPO_ROOT}/cluster/requirements-hpc.txt}"
PYTHON="${VENV_DIR}/bin/python"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "Apptainer is not available on PATH." >&2
    exit 2
fi
if [[ ! -d "${DAIC_STORAGE_ROOT}" ]]; then
    echo "DAIC storage root does not exist: ${DAIC_STORAGE_ROOT}" >&2
    echo "Set DAIC_STORAGE_ROOT to a persistent writable home or project directory." >&2
    exit 2
fi
if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Requirements file does not exist: ${REQUIREMENTS_FILE}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${IMAGE}")" "$(dirname "${VENV_DIR}")" "${APPTAINER_ROOT}/cache" "${APPTAINER_ROOT}/tmp"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_ROOT}/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${APPTAINER_ROOT}/tmp}"

if [[ ! -f "${IMAGE}" ]]; then
    echo "Pulling ${IMAGE_SOURCE}"
    echo "Image destination: ${IMAGE}"
    apptainer pull "${IMAGE}" "${IMAGE_SOURCE}"
fi

RUNTIME=(
    apptainer exec
    --bind "${REPO_ROOT}:${REPO_ROOT}"
    --bind "${DAIC_STORAGE_ROOT}:${DAIC_STORAGE_ROOT}"
    "${IMAGE}"
)

if ! "${RUNTIME[@]}" test -x "${PYTHON}"; then
    "${RUNTIME[@]}" python -m venv --system-site-packages "${VENV_DIR}"
fi

"${RUNTIME[@]}" "${PYTHON}" -m pip install \
    pip==23.3.2 setuptools==66.1.1 wheel==0.42.0
"${RUNTIME[@]}" "${PYTHON}" -m pip install \
    -r "${REQUIREMENTS_FILE}"
"${RUNTIME[@]}" "${PYTHON}" "${REPO_ROOT}/setup.py" build_ext --inplace

"${RUNTIME[@]}" "${PYTHON}" - <<'PY'
import sys
import torch
import numpy
import ding
import gym
from lzero.entry.train_ensemble_muzero import train_ensemble_muzero

print('python', sys.version.split()[0])
print('torch', torch.__version__, 'torch_cuda', torch.version.cuda)
print('numpy', numpy.__version__)
print('ding', getattr(ding, '__version__', 'unknown'))
print('gym', gym.__version__)
print('training_entry_import', train_ensemble_muzero.__name__)
print('cuda_available_on_login_node', torch.cuda.is_available())
PY

echo "apptainer_image=${IMAGE}"
echo "requirements_file=${REQUIREMENTS_FILE}"
echo "environment_ready=${VENV_DIR}"
