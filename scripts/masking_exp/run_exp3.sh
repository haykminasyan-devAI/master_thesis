#!/bin/bash
#SBATCH --job-name=dust3r_exp3
#SBATCH --partition=a100
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp3_%j.log
#SBATCH --error=logs/exp3_%j.err

# ── Configuration ─────────────────────────────────────────────────
SEQ_DIR="data/co3d/teddybear/101_11758_21048"
MASKED_DIR="outputs/masked_frames/teddybear/101_11758_21048/mask_25pct"
OUTPUT_DIR="outputs/dust3r/exp3"
EXP4_DIR="outputs/dust3r/exp4"
N_FRAMES_MIN=2
N_FRAMES_MAX=40
MASK_RATIO=0.25

# ── Environment ───────────────────────────────────────────────────
source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Sequence   : $SEQ_DIR"
echo "n_frames   : $N_FRAMES_MIN → $N_FRAMES_MAX  (ALL masked)"
echo "mask_ratio : $MASK_RATIO  |  num_patches: 3"
echo "================================================================"

echo ""
echo ">>> Step 1: Skipped — masked frames already in $MASKED_DIR"

# ── Step 2: Run inference in parallel (3 jobs at a time) ──────────
echo ""
echo ">>> Step 2: Running DUSt3R inference sweep (mask=25%) ..."

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
        --masked_dir   "$MASKED_DIR" \
        --output_dir   "$OUT_SUBDIR" \
        --mask_ratio   "$MASK_RATIO" \
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

# Wait for remaining jobs
if [ ${#ACTIVE_PIDS[@]} -gt 0 ]; then
    echo "  Waiting for final batch: n=${ACTIVE_N[*]} ..."
    for i in "${!ACTIVE_PIDS[@]}"; do
        wait "${ACTIVE_PIDS[$i]}"
        STATUS=$?
        [ $STATUS -eq 0 ] && echo "    [OK]   n=${ACTIVE_N[$i]}" \
                          || echo "    [FAIL] n=${ACTIVE_N[$i]} (exit $STATUS)"
    done
fi

# ── Step 3: Recompute metrics for exp4 (add new metrics to existing PLY) ──
echo ""
echo ">>> Step 3: Recomputing metrics for Exp4 (adding Hausdorff, F1, PSNR d1) ..."
python3 scripts/masking_exp/recompute_metrics.py \
    --exp_dir "$EXP4_DIR" \
    --gt_ply  "${SEQ_DIR}/pointcloud.ply"

# ── Step 4: Plot results ───────────────────────────────────────────
echo ""
echo ">>> Step 4: Generating plots ..."
python3 scripts/masking_exp/exp3.py \
    --sequence_dir  "$SEQ_DIR" \
    --masked_dir    "$MASKED_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --exp4_dir      "$EXP4_DIR" \
    --n_frames_min  "$N_FRAMES_MIN" \
    --n_frames_max  "$N_FRAMES_MAX" \
    --mask_ratio    "$MASK_RATIO" \
    --plot_only

echo ""
echo "================================================================"
echo "Job finished: $(date)"
echo "Plots:"
echo "  $OUTPUT_DIR/chamfer_vs_n_frames.png  (Exp3 only)"
echo "  $OUTPUT_DIR/exp3_vs_exp4.png         (comparison)"
echo "================================================================"
