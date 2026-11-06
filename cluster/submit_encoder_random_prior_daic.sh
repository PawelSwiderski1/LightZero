#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/submit_daic.sh" \
    experiments/run_mountain_car_encoder_random_prior.py \
    "$@"
