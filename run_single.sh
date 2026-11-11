#!/usr/bin/env bash
set -euo pipefail
RISDA_CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${1:-1}"
if [[ $# -gt 0 ]]; then
    shift
fi
cd "$RISDA_CHECKOUT"
exec "${RISDA_PYTHON:-python}" -m risda --gpus "$GPUS" "$@"
