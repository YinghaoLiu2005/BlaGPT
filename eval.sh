cd /data/yinghaoliu/BlaGPT
export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --standalone --nproc_per_node=4 bla_gpt/eval_mamba_checkpoints.py \
  --lengths 1024 2048 4096 8192 \
  --output_csv logs/mamba_25000_length_eval_02.csv\
  --device_batch_size 4 2>&1 | tee eval.txt
