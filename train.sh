export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nproc_per_node=2 bla_gpt/train.py --run_name mamba_sliding_4_15 --model_name blagpt  #--resume /data/yinghaoliu/BlaGPT/logs/experiment_4_01_kernel_1/best_model_25000.pt