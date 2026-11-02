#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -u -m scripts.eval_re10k2acid "$@"
