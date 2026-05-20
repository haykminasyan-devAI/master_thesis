#!/bin/bash
#SBATCH --job-name=dust3r_cmp
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/compare_demo_%j.log
#SBATCH --error=logs/compare_demo_%j.err

PORT=7870

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "Starting comparison demo on port $PORT ..."
echo "SSH tunnel: ssh -L ${PORT}:localhost:${PORT} <your_cluster_address>"

python3 scripts/compare_demo.py --port $PORT
