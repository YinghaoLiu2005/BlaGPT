#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${CONDA_PREFIX:-}" != "/data/yinghaoliu/conda/conda_envs/fla" ]]; then
  echo "Please run: conda activate /data/yinghaoliu/conda/conda_envs/fla" >&2
  exit 1
fi

export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

"/data/yinghaoliu/conda/conda_envs/fla/bin/torchrun" --standalone --nproc_per_node="$NPROC_PER_NODE" \
  bla_gpt/eval_mamba_checkpoints.py \
  --lengths 1024 2048 4096 8192 \
  --output_csv logs/mamba_25000_length_eval_02.csv \
  --device_batch_size 4 2>&1 | tee eval.txt
