export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nproc_per_node=2 bla_gpt/train.py --run_name mamba_sliding_test_change --model_name blagpt  #--resume /data/yinghaoliu/BlaGPT/logs/mamba_sliding_test_4/best_model_2750.pt