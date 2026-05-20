#!/bin/bash
#SBATCH --job-name=colmap_sweep
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/colmap_%x_%j.log
#SBATCH --error=logs/colmap_%x_%j.err

# ── Set by submit_colmap_all.sh ───────────────────────────────────────────────
# CATEGORY  e.g. teddybear
# SEQ_ID    e.g. 101_11758_21048
# GPU_IDX   e.g. 0

CATEGORY="${CATEGORY:-teddybear}"
SEQ_ID="${SEQ_ID:-101_11758_21048}"
GPU_IDX="${GPU_IDX:-0}"

MASK_RATIO=0.25
NUM_PATCHES=3

SEQ_DIR="data/co3d/${CATEGORY}/${SEQ_ID}"
IMAGES_DIR="${SEQ_DIR}/images"
MASKED_DIR="outputs/masked_frames/${CATEGORY}/${SEQ_ID}/mask_25pct"
OUTPUT_ROOT="outputs/colmap/masked_25pct/${CATEGORY}_${SEQ_ID}"

# ── Sparse sweep: dense at small n, sparse at larger n ────────────────────────
# Total: 21 values  →  ~3-5 hours on 1 GPU
FRAMES_TO_TRY=(2 3 4 5 6 7 8 9 10 15 20 25 30 40 50 60 70 80 90 100)

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
export TORCH_CUDA_ARCH_LIST="8.0"
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Category     : $CATEGORY  /  $SEQ_ID"
echo "GPU index    : $GPU_IDX"
echo "================================================================"

# ── Step 0: Generate masked frames (skip if already done) ─────────────────────
if [ -d "$MASKED_DIR" ] && [ "$(ls -A $MASKED_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
    echo ">>> [skip] masked frames already exist: $MASKED_DIR"
else
    echo ">>> Generating masked frames for ${CATEGORY}/${SEQ_ID} ..."
    python scripts/masking_exp/mask.py \
        --images_dir  "$IMAGES_DIR" \
        --output_dir  "$MASKED_DIR" \
        --mask_ratio  $MASK_RATIO \
        --num_patches $NUM_PATCHES \
        --seed        42
    echo "    Done generating masked frames."
fi

# ── Step 1: Check how many frames are available ────────────────────────────────
MAX_AVAIL=$(ls "$MASKED_DIR" 2>/dev/null | grep -Ec '\.(jpg|jpeg|png)$')
echo ""
echo ">>> Available masked frames: $MAX_AVAIL"

# Filter FRAMES_TO_TRY to only include values <= MAX_AVAIL
FRAMES_FILTERED=()
for N in "${FRAMES_TO_TRY[@]}"; do
    if [ "$N" -le "$MAX_AVAIL" ]; then
        FRAMES_FILTERED+=("$N")
    fi
done
echo ">>> Will run COLMAP for n = ${FRAMES_FILTERED[*]}"
echo ""

# ── Step 2: Run COLMAP for each n ─────────────────────────────────────────────
for N in "${FRAMES_FILTERED[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/frames_$(printf '%02d' $N)"

    if [ -f "${OUT_DIR}/metrics.txt" ]; then
        echo "  [skip] n=${N} — metrics.txt already exists"
        continue
    fi

    echo "  --- n=${N} / max=${MAX_AVAIL} ---"
    python scripts/expCOLMAP/run_colmap_inference.py \
        --masked_dir   "$MASKED_DIR" \
        --sequence_dir "$SEQ_DIR" \
        --output_dir   "$OUT_DIR" \
        --n_frames     $N \
        --gpu_index    $GPU_IDX

    # show result
    if [ -f "${OUT_DIR}/metrics.txt" ]; then
        STATUS=$(grep "^status" "${OUT_DIR}/metrics.txt" | awk '{print $2}')
        CD=$(grep "^chamfer_distance" "${OUT_DIR}/metrics.txt" | awk '{print $2}')
        echo "  n=${N} → status=${STATUS}  chamfer=${CD:-N/A}"
    fi
    echo ""
done

echo "================================================================"
echo "All n_frames complete for ${CATEGORY}/${SEQ_ID}"
echo "Results in: ${OUTPUT_ROOT}"
echo "Job finished : $(date)"
echo "================================================================"
