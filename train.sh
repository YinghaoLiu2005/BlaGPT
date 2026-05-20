export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAMBA_WINDOW_SIZE=${MAMBA_WINDOW_SIZE:-512}
RUN_NAME=${RUN_NAME:-mamba_sliding_window_attention_w${MAMBA_WINDOW_SIZE}}

torchrun --standalone --nproc_per_node=4 bla_gpt/train.py --run_name "$RUN_NAME" --model_name blagpt --attention mamba2_sliding_test --mamba_window_size $MAMBA_WINDOW_SIZE #--resume /data/yinghaoliu/BlaGPT/logs/mamba_sliding_test_4/best_model_2750.pt
#torchrun --standalone --nproc_per_node=4 bla_gpt/train.py --run_name regular --model_name blagpt
