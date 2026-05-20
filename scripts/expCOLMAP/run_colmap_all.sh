#!/bin/bash
#SBATCH --job-name=colmap_all
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/colmap_all_%j.log
#SBATCH --error=logs/colmap_all_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

# COLMAP works best with many frames — sweep 50..150 in steps of 10
FRAMES_TO_TRY=(50 60 70 80 90 100 110 120 130 140 150)

MASK_RATIO=0.25
NUM_PATCHES=3
N_GPUS=4
SEQ_BASE="data/co3d"
OUTPUT_ROOT="outputs/colmap/masked_25pct"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
export TORCH_CUDA_ARCH_LIST="8.0"
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Sequences    : ${!SEQUENCES[*]}"
echo "n_frames     : ${FRAMES_TO_TRY[*]}"
echo "GPUs         : ${N_GPUS}"
echo "================================================================"

# ── Step 1: Generate masked frames for all 6 categories (CPU, parallel) ───────
echo ""
echo ">>> Step 1: Generating masked frames ..."

MASK_PIDS=()
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    IMAGES_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images"
    MASKED_DIR="outputs/masked_frames/${CATEGORY}/${SEQ_ID}/mask_25pct"

    if [ -d "$MASKED_DIR" ] && [ "$(ls -A $MASKED_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [skip] ${CATEGORY} masked frames already exist"
    else
        echo "  Generating: ${CATEGORY}/${SEQ_ID} ..."
        python scripts/masking_exp/mask.py \
            --images_dir  "$IMAGES_DIR" \
            --output_dir  "$MASKED_DIR" \
            --mask_ratio  $MASK_RATIO \
            --num_patches $NUM_PATCHES \
            --seed        42 &
        MASK_PIDS+=($!)
    fi
done

# Wait for all mask generation to finish
for PID in "${MASK_PIDS[@]}"; do
    wait "$PID"
done
echo ">>> All masked frames ready."

# ── Step 2: Build work queue of all (category, n_frames) tasks ────────────────
echo ""
echo ">>> Step 2: Building work queue ..."

WORK_QUEUE=()
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    MASKED_DIR="outputs/masked_frames/${CATEGORY}/${SEQ_ID}/mask_25pct"

    # Only include n values that don't exceed available frame count
    MAX_AVAIL=$(ls "$MASKED_DIR" 2>/dev/null | grep -Ec '\.(jpg|jpeg|png)$')

    for N in "${FRAMES_TO_TRY[@]}"; do
        if [ "$N" -gt "$MAX_AVAIL" ]; then
            continue
        fi
        OUT_DIR="${OUTPUT_ROOT}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"
        if grep -q "status:.*ok" "${OUT_DIR}/metrics.txt" 2>/dev/null; then
            continue   # already done successfully
        fi
        WORK_QUEUE+=("${CATEGORY}|${SEQ_ID}|${N}")
    done
done

TOTAL=${#WORK_QUEUE[@]}
echo ">>> Total tasks to run: ${TOTAL}  (across ${N_GPUS} GPUs)"

# ── Step 3: Dispatch tasks across 4 GPUs ──────────────────────────────────────
echo ""
echo ">>> Step 3: Dispatching tasks ..."

# GPU slot tracking: GPU_PID[i] = PID of current task on GPU i (0 = idle)
declare -a GPU_PID
for (( i=0; i<N_GPUS; i++ )); do GPU_PID[$i]=0; done

GPU_POOL=(0 1 2 3)
TASK_IDX=0
DONE=0

while [ $DONE -lt $TOTAL ]; do
    for GPU_IDX in "${GPU_POOL[@]}"; do
        PID=${GPU_PID[$GPU_IDX]}

        # Check if this GPU slot is free
        if [ "$PID" -ne 0 ] && kill -0 "$PID" 2>/dev/null; then
            continue   # still running
        fi

        # Slot is free — mark as idle
        GPU_PID[$GPU_IDX]=0

        # If there's a task left, dispatch it
        if [ $TASK_IDX -lt $TOTAL ]; then
            TASK="${WORK_QUEUE[$TASK_IDX]}"
            IFS='|' read -r CATEGORY SEQ_ID N <<< "$TASK"

            SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"
            MASKED_DIR="outputs/masked_frames/${CATEGORY}/${SEQ_ID}/mask_25pct"
            OUT_DIR="${OUTPUT_ROOT}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"

            echo "  [GPU ${GPU_IDX}] task $((TASK_IDX+1))/${TOTAL}: ${CATEGORY} n=${N}"

            # Create output dir before redirect so the log file path is valid
            mkdir -p "$OUT_DIR"

            python scripts/expCOLMAP/run_colmap_inference.py \
                --masked_dir   "$MASKED_DIR" \
                --sequence_dir "$SEQ_DIR" \
                --output_dir   "$OUT_DIR" \
                --n_frames     "$N" \
                --gpu_index    "$GPU_IDX" \
                > "${OUT_DIR}/run.log" 2>&1 &

            GPU_PID[$GPU_IDX]=$!
            TASK_IDX=$((TASK_IDX + 1))
        fi
    done

    # Count finished tasks
    DONE=0
    for (( i=0; i<N_GPUS; i++ )); do
        if [ "${GPU_PID[$i]}" -eq 0 ] || ! kill -0 "${GPU_PID[$i]}" 2>/dev/null; then
            GPU_PID[$i]=0
        fi
    done
    running=$(( $(for p in "${GPU_PID[@]}"; do echo $p; done | grep -cv '^0$') ))
    DONE=$(( TASK_IDX - running ))

    if [ $TASK_IDX -lt $TOTAL ] || [ $running -gt 0 ]; then
        sleep 5
    fi
done

# Wait for all remaining background tasks
for (( i=0; i<N_GPUS; i++ )); do
    if [ "${GPU_PID[$i]}" -ne 0 ]; then
        wait "${GPU_PID[$i]}"
    fi
done

echo ""
echo ">>> All ${TOTAL} tasks complete."

# ── Step 4: Generate plot ──────────────────────────────────────────────────────
echo ""
echo ">>> Step 4: Generating plot ..."
python scripts/expCOLMAP/plot_colmap_results.py \
    --results_root "$OUTPUT_ROOT" \
    --output_dir   "$OUTPUT_ROOT"

echo ""
echo "================================================================"
echo "Job finished : $(date)"
echo "Results in   : ${OUTPUT_ROOT}"
echo "Plot         : ${OUTPUT_ROOT}/colmap_chamfer_vs_nframes.png"
echo "================================================================"
