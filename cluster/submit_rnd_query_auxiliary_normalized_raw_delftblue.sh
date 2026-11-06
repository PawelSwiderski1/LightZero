#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${LIGHTZERO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIGS="shared_normalized_raw member_normalized_raw" \
    "${REPO_ROOT}/cluster/submit_rnd_query_auxiliary_decoder_ablation_delftblue.sh"
