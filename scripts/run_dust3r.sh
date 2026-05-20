#!/bin/bash
#SBATCH --job-name=dust3r_inference
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/dust3r_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/dust3r_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env

export TORCH_CUDA_ARCH_LIST="8.0"

cd /home/asds/project_Hayk_Minasyan

python scripts/run_dust3r_inference.py \
    --sequence_dir data/co3d/teddybear/101_11758_21048 \
    --dust3r_dir dust3r \
    --n_frames 10 \
    --output_dir outputs/dust3r/teddybear_101_11758_21048_10frames \
    --device cuda

echo "Job finished: $(date)"
