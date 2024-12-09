#!/bin/bash
#SBATCH --job-name=flan-t5-task     # Tên job
#SBATCH -o flan-t5-task.out         # Tên file log xuất ra
#SBATCH --gres=gpu:1                # Yêu cầu tài nguyên GPU (1 GPU)
#SBATCH -N 1                        # Số lượng node sử dụng
#SBATCH --cpus-per-task=2           # Số lượng CPU trên mỗi task
#SBATCH --mem=50G                   # Bộ nhớ RAM cần dùng (50GB)

# Chọn GPU số 0
export CUDA_VISIBLE_DEVICES=4 

# Load module Python 3.9 hoặc môi trường Python bạn cần
module load python39                 

# Tạo môi trường ảo nếu chưa có
if [ ! -d "env" ]; then
  python3 -m venv env
fi

# Kích hoạt môi trường ảo
source env/bin/activate

# Cập nhật pip và cài thư viện từ requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# Chạy mã Python chính với tham số cụ thể
python main.py --model_name google/flan-t5-xxl \
               --test_data_path test_data.json \
               --output_path results-flan-t5.json
# Tắt virtual environment (chỉ nếu đã kích hoạt từ trước)
deactivate