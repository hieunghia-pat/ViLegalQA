export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export CUDA_VISIBLE_DEVICES=0

python3 task-1-2.py /home/ngiangh/.llama/checkpoints/Llama3.2-3B-Instruct
