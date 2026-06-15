#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${CONDA_PREFIX:-}" != "/data/yinghaoliu/conda/conda_envs/fla" ]]; then
  echo "Please run: conda activate /data/yinghaoliu/conda/conda_envs/fla" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

MAMBA_WINDOW_SIZE=512


for ATTENTION in kda
do
  RUN_NAME=${ATTENTION}_w${MAMBA_WINDOW_SIZE}
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" bla_gpt/train.py \
    --run_name "$RUN_NAME" \
    --model_name blagpt \
    --attention "$ATTENTION" \
    --num_epochs "$NUM_EPOCHS"

done
