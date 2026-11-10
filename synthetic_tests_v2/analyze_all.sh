#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
engine="${MAMBA_PYTHON:-python}"
"$engine" mqar_analysis.py --out_root out/mqar --save_dir out/analysis/mqar
# KLA's original plotter defaults exclude pf1, although its launcher enables it.
"$engine" palindrome_analysis.py --out_root out/palindrome --save_dir out/analysis/palindrome \
  --include_experiment '' --exclude_path_keywords '_backup_'
"$engine" stack_analysis.py --out_root out/stack --save_dir out/analysis/stack
