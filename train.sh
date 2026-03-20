export PYTHONPATH=/data/yinghaoliu/BlaGPT/mamba:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nproc_per_node=2 bla_gpt/train.py --run_name my_experiment --model_name blagpt