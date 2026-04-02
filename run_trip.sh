#!/bin/bash
#SBATCH --job-name=trip_attack_eps=100
#SBATCH --output=logs/mps_%j.out
#SBATCH --error=logs/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:2
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
REQUIRED_VRAM=30720
# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack
echo "ENV:" $CONDA_DEFAULT_ENV
echo "PREFIX:" $CONDA_PREFIX
which python
python -c "import sys; print(sys.executable)"

unset CUDA_VISIBLE_DEVICES
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

python run_all_batch.py \
    --runned_data_path model_evaluation_results/vgg16_selection.json \
    --output_dir output_trip_eps=100/vgg16_inter \
    --model_name vgg16 \
    --eps 100 \
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
    --attack_algo tripoaa \


python run_all_batch.py \
    --runned_data_path model_evaluation_results/vgg16_selection.json \
    --output_dir output_trip_eps=100/vgg16_simple \
    --model_name vgg16 \
    --eps 100 \
    --iterations 200 \
    --pc 0.1 \
    --pm 0.4 \
    --pop_size 100 \
    --zero_probability 0.3 \
    --max_dist 1e-5 \
    --p_size 2.0 \
    --tournament_size 4 \
    --explain_method simple_grad \
    --ig_steps 5 \
    --max_samples 100 \
    --attack_algo tripoaa \


python run_all_batch.py \
    --runned_data_path model_evaluation_results/resnet18_selection.json \
    --output_dir output_trip_eps=100/resnet18_simple \
    --model_name resnet18 \
    --eps 100 \
    --iterations 200 \
    --pc 0.1 \
    --pm 0.4 \
    --pop_size 100 \
    --zero_probability 0.3 \
    --max_dist 1e-5 \
    --p_size 2.0 \
    --tournament_size 4 \
    --explain_method simple_grad \
    --ig_steps 5 \
    --max_samples 100 \
    --attack_algo tripoaa \

python run_all_batch.py \
    --runned_data_path model_evaluation_results/resnet18_selection.json \
    --output_dir output_trip_eps=100/resnet18_inter \
    --model_name resnet18 \
    --eps 100 \
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
    --attack_algo tripoaa \


python run_all_batch.py \
    --runned_data_path model_evaluation_results/densenet121_selection.json \
    --output_dir output_trip_eps=100/densenet121_inter \
    --model_name densenet121 \
    --eps 100 \
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
    --attack_algo tripoaa \

python run_all_batch.py \
    --runned_data_path model_evaluation_results/densenet121_selection.json \
    --output_dir output_trip_eps=100/densenet121_simple \
    --model_name densenet121 \
    --eps 100 \
    --iterations 200 \
    --pc 0.1 \
    --pm 0.4 \
    --pop_size 100 \
    --zero_probability 0.3 \
    --max_dist 1e-5 \
    --p_size 2.0 \
    --tournament_size 4 \
    --explain_method simple_grad \
    --ig_steps 5 \
    --max_samples 100 \
    --attack_algo tripoaa \

python run_all_batch.py \
    --runned_data_path model_evaluation_results/vit_b_32_selection.json \
    --output_dir output_trip_eps=100/vit_b_32_inter \
    --model_name vit_b_32 \
    --eps 100 \
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
    --attack_algo tripoaa \

python run_all_batch.py \
    --runned_data_path model_evaluation_results/vit_b_32_selection.json \
    --output_dir output_trip_eps=100/vit_b_32_simple \
    --model_name vit_b_32 \
    --eps 100 \
    --iterations 200 \
    --pc 0.1 \
    --pm 0.4 \
    --pop_size 100 \
    --zero_probability 0.3 \
    --max_dist 1e-5 \
    --p_size 2.0 \
    --tournament_size 4 \
    --explain_method simple_grad \
    --ig_steps 5 \
    --max_samples 100 \
    --attack_algo tripoaa \
