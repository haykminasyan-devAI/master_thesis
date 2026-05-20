#!/bin/bash
#SBATCH --job-name=dust3r_demo
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/demo_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/demo_%j.err

# ── configure here ────────────────────────────────────────────────────────────
SEQUENCE_DIR="data/co3d/teddybear/101_11758_21048"
N_FRAMES=10
PORT=7860
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Sequence    : $SEQUENCE_DIR"
echo "Frames      : $N_FRAMES   Port: $PORT"
echo "================================================================"
echo ""
echo ">>> On your LOCAL machine run:"
echo "    ssh -L ${PORT}:$(hostname):${PORT} asds@<login-node-ip>"
echo ">>> Then open:  http://localhost:${PORT}"
echo ""

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
export TORCH_CUDA_ARCH_LIST="8.0"
cd /home/asds/project_Hayk_Minasyan

python scripts/auto_demo.py \
    --sequence_dir "$SEQUENCE_DIR" \
    --n_frames     "$N_FRAMES" \
    --port         "$PORT" \
    --device       cuda

echo "Job finished: $(date)"
