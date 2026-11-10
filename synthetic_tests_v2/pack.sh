#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
destination="${1:-$root/synthetic_tests_v2.tar.gz}"
base="$(basename "$root")"
archive_tmp="$(mktemp "${TMPDIR:-/tmp}/synthetic-tests-v2.XXXXXX.tar.gz")"
trap 'rm -f -- "$archive_tmp"' EXIT
tar --exclude="$base/data" \
    --exclude="$base/out" \
    --exclude="$base/test_outputs" \
    --exclude="$base/.venvs" \
    --exclude="$base/wandb" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.tar.gz' \
    -czf "$archive_tmp" -C "$(dirname "$root")" "$base"
mv -- "$archive_tmp" "$destination"
sha256sum "$destination"
