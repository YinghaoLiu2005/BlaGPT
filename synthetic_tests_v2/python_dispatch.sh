#!/usr/bin/env bash
set -euo pipefail
engine="${MAMBA_PYTHON:-python}"
previous=''
for argument in "$@"; do
  if [[ "$previous" == '--model_name' ]]; then
    case "$argument" in
      KDA_*|GDN_*) engine="${FLA_PYTHON:-python}" ;;
    esac
  fi
  previous="$argument"
done
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "$engine" "$@"
  printf '\n'
  exit 0
fi
# Offline logs preserve the original logger without requiring an account.
export WANDB_MODE="${WANDB_MODE:-offline}"
exec "$engine" "$@"
