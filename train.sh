export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 bla_gpt/train.py --run_name experiment_4_05_kernel --model_name blagpt  #--resume /data/yinghaoliu/BlaGPT/logs/experiment_4_01_kernel_1/best_model_25000.pt