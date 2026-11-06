#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PDEP="/data/yinghaoliu/cache/uv/archive-v0/XJyYec6VabYR1Q40:/data/yinghaoliu/cache/uv/archive-v0/kVu9aRc1YqjvqZlk:/data/yinghaoliu/cache/uv/archive-v0/QhR_OFg--t0l0bhx:/data/yinghaoliu/cache/uv/archive-v0/NvfKYmlMU5DwnAIH"
export PYTHONPATH="$PDEP:$ROOT:$ROOT/3rdparty/flash-linear-attention:${PYTHONPATH:-}"
# Keep three independent GPU workers from spawning 32 compiler processes each.
# The cache is shared by all workers so the first kernel build can be reused.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/output/kda_mqar_run/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/output/kda_mqar_run/torchinductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

run_phase() {
  local label="$1" env_name="$2" module_name="$3"
  local py="/data/yinghaoliu/conda/conda_envs/${env_name}/bin/python"
  local logdir="$ROOT/output/kda_mqar_run/${label}"
  mkdir -p "$logdir"
  echo "[$label] launching 3 GPU partitions with $env_name"
  local pids=()
  for gpu in 0 1 2; do
    CUDA_VISIBLE_DEVICES="$gpu" "$py" "$ROOT/scripts/run_partition.py" \
      "$module_name" "$gpu" 3 >"$logdir/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local rc=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  return "$rc"
}

run_phase fla fla zoology.experiments.kda_mqar.configs_fla
run_phase blagpt blagpt zoology.experiments.kda_mqar.configs_blagpt
