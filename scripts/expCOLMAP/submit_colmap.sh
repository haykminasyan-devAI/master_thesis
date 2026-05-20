#!/bin/bash
#SBATCH --job-name=colmap_sparse_test
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/colmap_sparse_test_%j.log
#SBATCH --error=logs/colmap_sparse_test_%j.err

cd /home/asds/project_Hayk_Minasyan
mkdir -p logs

echo "=== Job Starting at $(date) ==="
echo "=== Node: $(hostname) ==="

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env

echo "=== Python: $(which python) ==="
echo "=== Conda env: $CONDA_DEFAULT_ENV ==="

# ── NEW OUTPUT DIRECTORY (change frames_100_v2) ──────────────────────────────
python scripts/expCOLMAP/run_colmap_inference_2.py \
    --masked_dir   outputs/masked_frames/teddybear/101_11758_21048/mask_25pct \
    --sequence_dir data/co3d/teddybear/101_11758_21048 \
    --output_dir   outputs/colmap/masked_25pct/teddybear_101_11758_21048/frames_100_v5 \
    --n_frames     100 \
    --gpu_index    0

echo "=== Job Finished at $(date) ==="