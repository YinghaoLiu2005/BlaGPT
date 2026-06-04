export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAMBA_WINDOW_SIZE=${MAMBA_WINDOW_SIZE:-512}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
RUN_NAME=${RUN_NAME:-mamba2_GQA_w${MAMBA_WINDOW_SIZE}}

torchrun --standalone --nproc_per_node=4 bla_gpt/train.py --run_name "$RUN_NAME" --model_name blagpt --attention mamba2_GQA --mamba_window_size $MAMBA_WINDOW_SIZE --num_epochs $NUM_EPOCHS
#torchrun --standalone --nproc_per_node=4 bla_gpt/train.py --run_name regular --model_name blagpt
