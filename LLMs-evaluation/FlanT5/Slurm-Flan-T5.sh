#!/bin/bash
#SBATCH --job-name=flan-t5-task     # Tên job
#SBATCH -o flan-t5-task.out         # Tên file log xuất ra
#SBATCH --gres=gpu:1                # Yêu cầu tài nguyên GPU (1 GPU)
#SBATCH -N 1                        # Số lượng node sử dụng
#SBATCH --cpus-per-task=2           # Số lượng CPU trên mỗi task
#SBATCH --mem=50G                   # Bộ nhớ RAM cần dùng (50GB)

# Chọn GPU số 0
export CUDA_VISIBLE_DEVICES=4 

# Tải module cần thiết
module load python/3.8                # Tải môi trường Python (có thể điều chỉnh theo phiên bản)

# Tạo và kích hoạt virtual environment (chỉ nếu cần)
python3 -m venv env
source env/bin/activate

# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt

# Chạy mã Python chính với tham số cụ thể
python main.py --test_data_path /data/npl/MRC/ViLegalQA/LLMs-evaluation/Flan-T5/test_data.json --output_path results-flan-t5.json

# Tắt virtual environment (chỉ nếu đã kích hoạt từ trước)
deactivate