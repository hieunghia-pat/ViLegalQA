#!/bin/sh
#SBATCH -o vinallama_train.out         # Tên file log output
#SBATCH --gres=gpu:1                   # Sử dụng 1 GPU
#SBATCH -N 1                           # Sử dụng 1 node
#SBATCH --ntasks=1                     # Tổng số task
#SBATCH --ntasks-per-node=1            # Số task mỗi node
#SBATCH --time=24:00:00                # Giới hạn thời gian chạy
#SBATCH --mem=32GB                     # Bộ nhớ tối đa

# Load module Python
module load python/3.9

# Cài đặt các gói từ requirements.txt (nếu chưa cài đặt)
pip install --user -r requirements.txt

# Chạy script Python
python3 main.py train \
  --dataset_path="test_data.json" \
  --epochs=50 \
  --batch_size=256 \
  --model="vinallama-7b-chat"