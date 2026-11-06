#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${LIGHTZERO_VENV:-${HOME}/.venvs/lightzero-pytorch-2.5.1-cu124}"
REQUIREMENTS_FILE="${LIGHTZERO_REQUIREMENTS_FILE:-${REPO_ROOT}/cluster/requirements-branching-rooms-daic.txt}"
PYTORCH_INDEX_URL="${LIGHTZERO_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PYTHON_MODULE="${LIGHTZERO_PYTHON_MODULE:-miniconda/3.10}"
BUILD_MODULE="${LIGHTZERO_BUILD_MODULE:-devtoolset/11}"
SETUP_TMP="${LIGHTZERO_SETUP_TMP:-/tmp/${USER}/lightzero-native-setup}"
PYTHON="${VENV_DIR}/bin/python"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Requirements file does not exist: ${REQUIREMENTS_FILE}" >&2
    exit 2
fi

module use /opt/insy/modulefiles
module load "${PYTHON_MODULE}" "${BUILD_MODULE}"

mkdir -p "$(dirname "${VENV_DIR}")" "${SETUP_TMP}/pip-cache" "${SETUP_TMP}/tmp"
export PIP_CACHE_DIR="${SETUP_TMP}/pip-cache"
export TMPDIR="${SETUP_TMP}/tmp"
export PIP_NO_CACHE_DIR=1

if [[ ! -x "${PYTHON}" ]]; then
    python -m venv "${VENV_DIR}"
fi

"${PYTHON}" -m pip install pip==23.3.2 setuptools==66.1.1 wheel==0.42.0
"${PYTHON}" -m pip install --index-url "${PYTORCH_INDEX_URL}" torch==2.5.1
"${PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"

cd "${REPO_ROOT}"
"${PYTHON}" setup.py build_ext --inplace

"${PYTHON}" - <<'PY'
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

rm -rf "${SETUP_TMP}"

echo "requirements_file=${REQUIREMENTS_FILE}"
echo "environment_ready=${VENV_DIR}"
du -sh "${VENV_DIR}"
