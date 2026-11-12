#!/usr/bin/env bash
set -euo pipefail
RISDA_CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:?Usage: bash run_train.sh STAGE [GPUS] [OPTIONS] [key=value ...]}"
shift
GPUS=1
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    GPUS="$1"
    shift
fi
cd "$RISDA_CHECKOUT"
exec "${RISDA_PYTHON:-python}" -m risda.train "$STAGE" --gpus "$GPUS" "$@"
