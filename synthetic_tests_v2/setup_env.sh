#!/usr/bin/env bash
# Usage: PYTHON=python3.10 bash setup_env.sh fla|mamba
set -euo pipefail
cd "$(dirname "$0")"
kind="${1:?Usage: bash setup_env.sh fla|mamba}"
case "$kind" in
  fla) torch_version=2.7.1; cuda_index=cu118 ;;
  mamba) torch_version=2.5.0; cuda_index=cu124 ;;
  *) echo "Expected fla or mamba" >&2; exit 2 ;;
esac
"${PYTHON:-python3}" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11), "Use Python 3.10 or 3.11"'
"${PYTHON:-python3}" -m venv ".venvs/$kind"
engine="$PWD/.venvs/$kind/bin/python"
"$engine" -m pip install --upgrade pip setuptools wheel ninja packaging
"$engine" -m pip install "torch==$torch_version" --index-url "https://download.pytorch.org/whl/$cuda_index"
"$engine" -m pip install --no-build-isolation -r "requirements-$kind.txt"
printf 'Created %s\n' "$PWD/.venvs/$kind"
