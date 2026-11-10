#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for task in mqar palindrome stack; do
  bash "run_${task}_experiment.sh"
done
