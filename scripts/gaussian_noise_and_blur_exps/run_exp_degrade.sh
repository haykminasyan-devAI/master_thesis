#!/bin/bash
#SBATCH --job-name=dust3r_degrade
#SBATCH --partition=h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp_degrade_%j.log
#SBATCH --error=logs/exp_degrade_%j.err

# ── Parameters (pass via --export when submitting) ────────────────────────────
#   DEGRADE_MODE  : "blur" or "noise"
#   DEGRADE_PARAM : blur sigma (float) or noise std (int), e.g. 3 or 25
#   CATEGORY      : CO3D category name, e.g. "teddybear"
#   SEQ_ID        : sequence folder name, e.g. "101_11758_21048"
#
# Example:
#   sbatch --export=ALL,DEGRADE_MODE=blur,DEGRADE_PARAM=3,CATEGORY=teddybear,SEQ_ID=101_11758_21048 \
#          scripts/gaussian_noise_and_blur_exps/run_exp_degrade.sh

: "${DEGRADE_MODE:?Please set DEGRADE_MODE (blur or noise)}"
: "${DEGRADE_PARAM:?Please set DEGRADE_PARAM (blur sigma or noise std)}"
: "${CATEGORY:?Please set CATEGORY (e.g. teddybear)}"
: "${SEQ_ID:?Please set SEQ_ID (e.g. 101_11758_21048)}"

SEQ_DIR="data/co3d/${CATEGORY}/${SEQ_ID}"
IMAGES_DIR="${SEQ_DIR}/images"
N_FRAMES_MIN=2
N_FRAMES_MAX=40

if [ "$DEGRADE_MODE" = "blur" ]; then
    TAG="blur_s${DEGRADE_PARAM}"
    DEGRADE_ARGS="--mode blur --blur_sigma ${DEGRADE_PARAM}"
else
    TAG="noise_s${DEGRADE_PARAM}"
    DEGRADE_ARGS="--mode noise --noise_std ${DEGRADE_PARAM} --seed 42"
fi

DEGRADED_DIR="outputs/degraded_frames/${CATEGORY}/${SEQ_ID}/${TAG}"
OUTPUT_DIR="outputs/dust3r/degrade_multi/${TAG}/${CATEGORY}_${SEQ_ID}"

# ── Environment ───────────────────────────────────────────────────────────────
source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Category     : ${CATEGORY}  seq=${SEQ_ID}"
echo "Mode         : $DEGRADE_MODE  param=$DEGRADE_PARAM  tag=$TAG"
echo "n_frames     : ${N_FRAMES_MIN} → ${N_FRAMES_MAX}"
echo "Output       : ${OUTPUT_DIR}"
echo "================================================================"

# ── Step 1: Generate degraded frames ─────────────────────────────────────────
echo ""
echo ">>> Step 1: Generating degraded frames ($TAG) ..."
if [ -d "$DEGRADED_DIR" ] && [ "$(ls -A $DEGRADED_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  [skip] $DEGRADED_DIR already exists and is non-empty"
else
    python3 scripts/gaussian_noise_and_blur_exps/degrade.py \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$DEGRADED_DIR" \
        $DEGRADE_ARGS
fi

# ── Step 2: DUSt3R inference sweep (3 GPUs, 3 at a time) ─────────────────────
echo ""
echo ">>> Step 2: Running DUSt3R inference sweep ..."
mkdir -p "$OUTPUT_DIR"

ACTIVE_PIDS=()
ACTIVE_N=()
GPU_POOL=(0 1 2 3)
NEXT_GPU=0

for N in $(seq $N_FRAMES_MIN $N_FRAMES_MAX); do
    if [ -f "${OUTPUT_DIR}/frames_$(printf '%02d' $N)/metrics.txt" ]; then
        echo "  [skip]  n_frames=$N"
        continue
    fi

    GPU_ID=${GPU_POOL[$NEXT_GPU]}
    OUT_SUBDIR="${OUTPUT_DIR}/frames_$(printf '%02d' $N)"
    mkdir -p "$OUT_SUBDIR"

    echo "  GPU $GPU_ID  →  n_frames=$N"

    PYTORCH_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 scripts/run_dust3r_inference.py \
        --sequence_dir "$SEQ_DIR" \
        --dust3r_dir   "dust3r" \
        --n_frames     "$N" \
        --n_masked     "$N" \
        --masked_dir   "$DEGRADED_DIR" \
        --output_dir   "$OUT_SUBDIR" \
        --mask_ratio   0.0 \
        > "${OUT_SUBDIR}/inference.log" 2>&1 &

    ACTIVE_PIDS+=($!)
    ACTIVE_N+=($N)
    NEXT_GPU=$(( (NEXT_GPU + 1) % 4 ))

    if [ ${#ACTIVE_PIDS[@]} -eq 4 ]; then
        echo "  Waiting for batch: n=${ACTIVE_N[*]} ..."
        for i in "${!ACTIVE_PIDS[@]}"; do
            wait "${ACTIVE_PIDS[$i]}"
            STATUS=$?
            [ $STATUS -eq 0 ] && echo "    [OK]   n=${ACTIVE_N[$i]}" \
                              || echo "    [FAIL] n=${ACTIVE_N[$i]} (exit $STATUS)"
        done
        ACTIVE_PIDS=()
        ACTIVE_N=()
        NEXT_GPU=0
    fi
done

if [ ${#ACTIVE_PIDS[@]} -gt 0 ]; then
    echo "  Waiting for final batch: n=${ACTIVE_N[*]} ..."
    for i in "${!ACTIVE_PIDS[@]}"; do
        wait "${ACTIVE_PIDS[$i]}"
        STATUS=$?
        [ $STATUS -eq 0 ] && echo "    [OK]   n=${ACTIVE_N[$i]}" \
                          || echo "    [FAIL] n=${ACTIVE_N[$i]} (exit $STATUS)"
    done
fi

# ── Step 3: Regenerate comparison plot (all experiments collected) ────────────
echo ""
echo ">>> Step 3: Updating comparison plot ..."
python3 scripts/gaussian_noise_and_blur_exps/plot_degrade_multi_seq.py \
    --results_root "outputs/dust3r/degrade_multi" \
    --n_min "$N_FRAMES_MIN" \
    --n_max "$N_FRAMES_MAX"

echo ""
echo "================================================================"
echo "Job finished : $(date)"
echo "Results      : ${OUTPUT_DIR}/"
echo "================================================================"
