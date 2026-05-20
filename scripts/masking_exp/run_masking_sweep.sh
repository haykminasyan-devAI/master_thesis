#!/bin/bash
#SBATCH --job-name=dust3r_sweep
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/sweep_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/sweep_%j.err

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
export TORCH_CUDA_ARCH_LIST="8.0"
cd /home/asds/project_Hayk_Minasyan

SEQ_DIR="data/co3d/teddybear/101_11758_21048"
MASKED_DIR="outputs/masked_frames/teddybear/101_11758_21048/mask_25pct"
MASK_RATIO=0.25
N_FRAMES=10

echo "================================================================"
echo "Job started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Sequence: $SEQ_DIR"
echo "Mask ratio: $MASK_RATIO  |  Total frames: $N_FRAMES"
echo "================================================================"

for N_MASKED in 0 1 2 3 4 5 6 7 8 9 10; do
    echo ""
    echo ">>> Running n_masked=${N_MASKED} ..."

    python scripts/run_dust3r_inference.py \
        --sequence_dir "$SEQ_DIR" \
        --dust3r_dir   dust3r \
        --n_frames     $N_FRAMES \
        --n_masked     $N_MASKED \
        --masked_dir   "$MASKED_DIR" \
        --mask_ratio   $MASK_RATIO \
        --output_dir   "outputs/dust3r/sweep/masked_${N_MASKED}of${N_FRAMES}" \
        --device       cuda

    echo "    Done. n_masked=${N_MASKED}"
done

echo ""
echo "================================================================"
echo "All runs complete. Results in outputs/dust3r/sweep/"
echo "Job finished: $(date)"
echo "================================================================"