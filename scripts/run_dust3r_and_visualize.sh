#!/bin/bash
#SBATCH --job-name=dust3r_viz
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/dust3r_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/dust3r_%j.err

# ── configurable parameters ───────────────────────────────────────────────────
SEQUENCE_DIR="data/co3d/teddybear/101_11758_21048"
N_FRAMES=10
OUTPUT_DIR="outputs/dust3r/teddybear_101_11758_21048_${N_FRAMES}frames"
VIZ_MODE="overlay"       # overlay | sidebyside
PORT=8080
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Sequence: $SEQUENCE_DIR  |  Frames: $N_FRAMES"
echo "================================================================"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
export TORCH_CUDA_ARCH_LIST="8.0"
cd /home/asds/project_Hayk_Minasyan

# ── Step 1: DUSt3R inference ──────────────────────────────────────────────────
echo ""
echo ">>> Step 1: Running DUSt3R inference ..."
python scripts/run_dust3r_inference.py \
    --sequence_dir "$SEQUENCE_DIR" \
    --dust3r_dir   dust3r \
    --n_frames     "$N_FRAMES" \
    --output_dir   "$OUTPUT_DIR" \
    --device       cuda

if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed. Check the .err log."
    exit 1
fi

# ── Step 2: Generate interactive HTML viewer ──────────────────────────────────
echo ""
echo ">>> Step 2: Generating interactive HTML viewer (mode: $VIZ_MODE) ..."

CATEGORY=$(echo "$SEQUENCE_DIR" | awk -F'/' '{{print $(NF-1)}}')
SEQ_NAME=$(basename "$SEQUENCE_DIR")
TITLE="${CATEGORY} · ${SEQ_NAME} · ${N_FRAMES} frames"
VIEWER_HTML="${OUTPUT_DIR}/viewer.html"

python scripts/visualize_pointclouds.py \
    --predicted "${OUTPUT_DIR}/predicted.ply" \
    --gt        "${SEQUENCE_DIR}/pointcloud.ply" \
    --output    "$VIEWER_HTML" \
    --title     "$TITLE" \
    --mode      "$VIZ_MODE"

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization failed."
    exit 1
fi

# ── Step 3: Start HTTP server ─────────────────────────────────────────────────
echo ""
echo ">>> Step 3: Starting HTTP server ..."
echo ""

NODE=$(hostname)
echo "================================================================"
echo "  Viewer ready!"
echo ""
echo "  On your LOCAL machine, run:"
echo "    ssh -L ${PORT}:${NODE}:${PORT} asds@<cluster_login_ip>"
echo ""
echo "  Then open in your browser:"
echo "    http://localhost:${PORT}/$(basename $VIEWER_HTML)"
echo "================================================================"

# serve the output directory (keeps running until job time limit)
cd "$OUTPUT_DIR"
python -m http.server "$PORT" --bind 0.0.0.0

echo "Job finished: $(date)"
