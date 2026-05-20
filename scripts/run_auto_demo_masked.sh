#!/bin/bash
#SBATCH --job-name=dust3r_demo
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/auto_demo_masked_%j.log
#SBATCH --error=logs/auto_demo_masked_%j.err

# ── Configuration ─────────────────────────────────────────────────
SEQUENCE_DIR="data/co3d/teddybear/101_11758_21048"
MASKED_DIR="outputs/masked_frames/teddybear/101_11758_21048/mask_25pct"
N_FRAMES=10
N_MASKED=3      # ← change to 4 to see n_masked=4
PORT=7860

# ── Environment ───────────────────────────────────────────────────
source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Sequence  : $SEQUENCE_DIR"
echo "n_frames  : $N_FRAMES  |  n_masked: $N_MASKED"
echo "Port      : $PORT"
echo "================================================================"
echo ""
echo "SSH tunnel (run on your local machine):"
echo "  ssh -L ${PORT}:localhost:${PORT} <your_cluster_address>"
echo "Then open: http://localhost:${PORT}"
echo ""

python3 scripts/auto_demo.py \
    --sequence_dir "$SEQUENCE_DIR" \
    --masked_dir   "$MASKED_DIR" \
    --n_frames     "$N_FRAMES" \
    --n_masked     "$N_MASKED" \
    --port         "$PORT"
