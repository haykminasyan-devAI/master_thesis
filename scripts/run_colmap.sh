#!/bin/bash
#SBATCH --job-name=colmap_exp
#SBATCH --output=logs/colmap_%j.log
#SBATCH --error=logs/colmap_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

echo "=========================================="
echo "  COLMAP Experiment – 20 frames, mask 25%"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPU    : $CUDA_VISIBLE_DEVICES"
echo "  Start  : $(date)"
echo "=========================================="

cd /home/asds/project_Hayk_Minasyan

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env

# ── run COLMAP ──────────────────────────────────────────────────────────────
python scripts/run_colmap.py \
    --images_dir   outputs/dust3r/exp2images/20 \
    --sequence_dir data/co3d/teddybear/101_11758_21048 \
    --output_dir   outputs/colmap/frames_20 \
    --n_frames     20 \
    --mask_ratio   0.25 \
    --gpu_index    0

echo ""
echo "=========================================="
echo "  Done: $(date)"
echo "=========================================="
