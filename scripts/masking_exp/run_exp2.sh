#!/bin/bash
#SBATCH --job-name=dust3r_exp2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp2_%j.log
#SBATCH --error=logs/exp2_%j.err

# ── Configuration ─────────────────────────────────────────────────
SEQ_DIR="data/co3d/teddybear/101_11758_21048"
MASKED_DIR="outputs/masked_frames/teddybear/101_11758_21048/mask_25pct"
OUTPUT_DIR="outputs/dust3r/exp2"
N_FRAMES_MIN=11
N_FRAMES_MAX=20
MASK_RATIO=0.25

# ── Environment ───────────────────────────────────────────────────
source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Sequence  : $SEQ_DIR"
echo "Masked dir: $MASKED_DIR"
echo "n_frames  : $N_FRAMES_MIN → $N_FRAMES_MAX  (ALL masked)"
echo "mask_ratio: $MASK_RATIO"
echo "================================================================"

python3 scripts/exp2.py \
    --sequence_dir  "$SEQ_DIR" \
    --masked_dir    "$MASKED_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --n_frames_min  "$N_FRAMES_MIN" \
    --n_frames_max  "$N_FRAMES_MAX" \
    --mask_ratio    "$MASK_RATIO"

echo ""
echo "================================================================"
echo "Job finished: $(date)"
echo "Results in : $OUTPUT_DIR"
echo "Plot       : $OUTPUT_DIR/chamfer_vs_n_frames.png"
echo "================================================================"
