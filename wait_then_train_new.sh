#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_new.sh"
POLL_SECONDS="${POLL_SECONDS:-30}"

usage() {
  echo "Usage: $0 [PID ...]"
  echo
  echo "Wait for the given PID(s), then immediately run train_new.sh."
  echo "If no PID is given, wait for currently running BlaGPT train_new.sh/torchrun training processes."
}

is_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

find_current_training_pids() {
  ps -eo pid=,cmd= | awk -v self="$$" '
    $1 == self { next }
    /awk -v self=/ { next }
    /wait_then_train_new\.sh/ { next }
    /bash train_new\.sh/ || (/torchrun/ && /bla_gpt\/train\.py/) {
      print $1
    }
  '
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$TRAIN_SCRIPT" ]]; then
  echo "Making $TRAIN_SCRIPT executable..."
  chmod +x "$TRAIN_SCRIPT"
fi

if [[ "$#" -gt 0 ]]; then
  pids=("$@")
else
  mapfile -t pids < <(find_current_training_pids)
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No running training process found. Starting train_new.sh now."
else
  echo "Waiting for PID(s): ${pids[*]}"
  while true; do
    alive=()
    for pid in "${pids[@]}"; do
      if is_alive "$pid"; then
        alive+=("$pid")
      fi
    done

    if [[ "${#alive[@]}" -eq 0 ]]; then
      break
    fi

    echo "$(date '+%F %T') still running: ${alive[*]}"
    sleep "$POLL_SECONDS"
  done
fi

echo "$(date '+%F %T') starting $TRAIN_SCRIPT"
cd "$SCRIPT_DIR"
exec "$TRAIN_SCRIPT"
