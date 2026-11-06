#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/3rdparty/flash-linear-attention:${PYTHONPATH:-}"
exec python -m zoology.launch "$ROOT/zoology/experiments/kda_mqar/configs.py" "$@"
