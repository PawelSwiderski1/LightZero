#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH:-/scratch/${USER}}"
APPTAINER_ROOT="${LIGHTZERO_APPTAINER_ROOT:-${SCRATCH_ROOT}/lightzero_apptainer}"
IMAGE_SOURCE="${LIGHTZERO_APPTAINER_SOURCE:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
IMAGE="${LIGHTZERO_APPTAINER_IMAGE:-${APPTAINER_ROOT}/images/pytorch-2.5.1-cuda12.4-cudnn9-devel.sif}"
VENV_DIR="${LIGHTZERO_VENV:-${APPTAINER_ROOT}/venvs/lightzero-pytorch-2.5.1}"
PYTHON="${VENV_DIR}/bin/python"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "Apptainer is not available on PATH." >&2
    exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${IMAGE}")" "$(dirname "${VENV_DIR}")" "${APPTAINER_ROOT}/cache"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_ROOT}/cache}"

if [[ ! -f "${IMAGE}" ]]; then
    echo "Pulling ${IMAGE_SOURCE}"
    echo "Image destination: ${IMAGE}"
    apptainer pull "${IMAGE}" "${IMAGE_SOURCE}"
fi

RUNTIME=(
    apptainer exec
    --bind "${REPO_ROOT}:${REPO_ROOT}"
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}"
    "${IMAGE}"
)

"${RUNTIME[@]}" python - <<'PY'
import sys

print('container_python', sys.version)
if sys.version_info < (3, 10):
    raise SystemExit('The selected container must provide Python >= 3.10')
PY

if ! "${RUNTIME[@]}" test -x "${PYTHON}"; then
    "${RUNTIME[@]}" python -m venv --system-site-packages "${VENV_DIR}"
fi

# DI-engine 0.5.3 requires setuptools <= 66.1.1. These versions also retain
# support for gym 0.25's legacy package metadata.
"${RUNTIME[@]}" "${PYTHON}" -m pip install \
    pip==23.3.2 setuptools==66.1.1 wheel==0.42.0
"${RUNTIME[@]}" "${PYTHON}" -m pip install \
    -r "${REPO_ROOT}/cluster/requirements-hpc.txt"

# Compile against the same Python, PyTorch, and system libraries used by jobs.
"${RUNTIME[@]}" "${PYTHON}" "${REPO_ROOT}/setup.py" build_ext --inplace

"${RUNTIME[@]}" "${PYTHON}" - <<'PY'
import sys
import torch
import numpy
import ding
import gym
import wandb
from google.protobuf import __version__ as protobuf_version
from lzero.entry.train_ensemble_muzero import train_ensemble_muzero

print('python', sys.version.split()[0])
print('torch', torch.__version__, 'torch_cuda', torch.version.cuda)
print('numpy', numpy.__version__)
print('ding', getattr(ding, '__version__', 'unknown'))
print('gym', gym.__version__)
print('wandb', wandb.__version__)
print('protobuf', protobuf_version)
print('training_entry_import', train_ensemble_muzero.__name__)
print('cuda_available_on_login_node', torch.cuda.is_available())
PY

echo "apptainer_image=${IMAGE}"
echo "environment_ready=${VENV_DIR}"
