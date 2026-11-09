#!/usr/bin/env bash
set -euo pipefail
RISDA_CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RISDA_CHECKOUT"
exec "${RISDA_PYTHON:-python}" -m risda run "$@"

