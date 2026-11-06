#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/submit_delftblue.sh" \
    experiments/run_mountain_car_corrected_rnd_intrinsic_diagnostic.py \
    "$@"
