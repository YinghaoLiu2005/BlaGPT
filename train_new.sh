#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_EPOCHS=${NUM_EPOCHS:-0}

MAMBA_WINDOW_SIZE=512


for ATTENTION in  mamba2
do
  RUN_NAME=${ATTENTION}_w${MAMBA_WINDOW_SIZE}
  torchrun --standalone --nproc_per_node=4 bla_gpt/train.py \
    --run_name "$RUN_NAME" \
    --model_name blagpt \
    --attention "$ATTENTION" \
    #--mamba_window_size "$MAMBA_WINDOW_SIZE"\
    --num_epochs "$NUM_EPOCHS"

done
