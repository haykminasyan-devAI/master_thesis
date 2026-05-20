#!/bin/bash
#SBATCH --job-name=dust3r_noise
#SBATCH --partition=a100
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp_noise_%j.log
#SBATCH --error=logs/exp_noise_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
SEQ_DIR="data/co3d/teddybear/101_11758_21048"
IMAGES_DIR="${SEQ_DIR}/images"
DEGRADED_DIR="outputs/degraded_frames/teddybear/101_11758_21048/noise_std25"
OUTPUT_DIR="outputs/dust3r/frames_0_40_noise_s25"
N_FRAMES_MIN=2
N_FRAMES_MAX=40
NOISE_STD=25

# ── Environment ───────────────────────────────────────────────────────────────
source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Experiment   : Gaussian Noise  std=${NOISE_STD} (0-255 scale)"
echo "n_frames     : ${N_FRAMES_MIN} → ${N_FRAMES_MAX}  (ALL frames degraded)"
echo "Output       : ${OUTPUT_DIR}"
echo "================================================================"

# ── Step 1: Generate noisy frames ─────────────────────────────────────────────
echo ""
echo ">>> Step 1: Applying Gaussian noise (std=${NOISE_STD}) to all frames ..."
if [ -d "$DEGRADED_DIR" ] && [ "$(ls -A $DEGRADED_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  [skip] $DEGRADED_DIR already exists and is non-empty"
else
    python3 scripts/gaussian_noise_and_blur_exps/degrade.py \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$DEGRADED_DIR" \
        --mode       noise \
        --noise_std  "$NOISE_STD" \
        --seed       42
fi

# ── Step 2: DUSt3R inference sweep (3 GPUs in parallel, 3 at a time) ─────────
echo ""
echo ">>> Step 2: Running DUSt3R inference sweep (noise, n=${N_FRAMES_MIN}..${N_FRAMES_MAX}) ..."

mkdir -p "$OUTPUT_DIR"

ACTIVE_PIDS=()
ACTIVE_N=()
GPU_POOL=(0 1 2)
NEXT_GPU=0

for N in $(seq $N_FRAMES_MIN $N_FRAMES_MAX); do
    if [ -f "${OUTPUT_DIR}/frames_$(printf '%02d' $N)/metrics.txt" ]; then
        echo "  [skip]  n_frames=$N (already done)"
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
    NEXT_GPU=$(( (NEXT_GPU + 1) % 3 ))

    if [ ${#ACTIVE_PIDS[@]} -eq 3 ]; then
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

# ── Step 3: Plot and compare ──────────────────────────────────────────────────
echo ""
echo ">>> Step 3: Generating comparison plots ..."
python3 scripts/gaussian_noise_and_blur_exps/plot_degrade_compare.py \
    --noise_dir  "$OUTPUT_DIR" \
    --mask25_dir "outputs/dust3r/frames_0_40_0.25_3" \
    --mask50_dir "outputs/dust3r/frames0_40_0,5_6" \
    --n_min      "$N_FRAMES_MIN" \
    --n_max      "$N_FRAMES_MAX" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "Job finished : $(date)"
echo "Results      : ${OUTPUT_DIR}/"
echo "================================================================"
