#!/bin/bash
#SBATCH --job-name=confidence_attack_eps=50
#SBATCH --output=logs/mps_%j.out
#SBATCH --error=logs/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:2 # Không khai báo GPU; --gres=mps:l40:2 (card L40); --gres=mps:a100:2 (card A100, ưu tiên các job dùng > 40GB vRAM)
#SBATCH --mem=4G
#SBATCH --time=14-00:00:00
REQUIRED_VRAM=31200  # Quan trọng - Số vRAM cần dùng (để tìm GPU phù hợp)
# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f
# *** Kích hoạt venv (Sửa đường dẫn / môi trường theo user)
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack
echo "ENV:" $CONDA_DEFAULT_ENV
echo "PREFIX:" $CONDA_PREFIX
which python
python -c "import sys; print(sys.executable)"

# Xóa biến môi trường Slurm để tự chọn GPU
unset CUDA_VISIBLE_DEVICES
# --- GỌI HELPER --- (Quan trọng, cần gọi hàm này (có sẵn) để tìm GPU có vRAM trống >= REQUIRED_VRAM, nếu không tìn thấy GPU đủ vRAM thì hàm CHECK_OUT sẽ đưa job vào lại hàng đợi để chờ tìm slot khác; sau 5 lần requeue mà vẫn chưa tìm được slot thì sẽ trả về mã lỗi để kết thúc job)
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0 
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1 
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"


export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU 

# =========================================================
# CHẠY CODE
# =========================================================



python run_GA_all_batch.py \
    --runned_data_path model_evaluation_results/vit_b_32_selection.json \
    --output_dir output_GA_eps=50/vit_b_32_inter \
    --model_name vit_b_32 \
    --eps 50 \
    --iterations 200 \
    --pc 0.1 \
    --pm 0.4 \
    --pop_size 100 \
    --zero_probability 0.3 \
    --max_dist 1e-5 \
    --p_size 2.0 \
    --tournament_size 4 \
    --explain_method integrated_gradients \
    --ig_steps 5 \
    --max_samples 100 \
    --lambda_1 1.0 \
    --lambda_2 1.0


