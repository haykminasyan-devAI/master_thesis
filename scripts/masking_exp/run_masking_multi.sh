#!/bin/bash
#SBATCH --job-name=mask_multi
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mask_multi_%j.log
#SBATCH --error=logs/mask_multi_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

# mask_tag | mask_ratio | num_patches
MASK_CONFIGS=(
    "mask_25pct|0.25|3"
    "mask_50pct|0.50|6"
)

N_FRAMES_MIN=2
N_FRAMES_MAX=20
N_GPUS=4
SEQ_BASE="data/co3d"

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
echo "Mask configs : mask_25pct(ratio=0.25,patches=3)  mask_50pct(ratio=0.50,patches=6)"
echo "n_frames     : ${N_FRAMES_MIN} → ${N_FRAMES_MAX}"
echo "GPUs         : ${N_GPUS}"
echo "================================================================"

# ── Step 1: Generate masked frames for all categories and both ratios ──────────
echo ""
echo ">>> Step 1: Generating masked frames ..."

MASK_PIDS=()
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    IMAGES_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images"

    for CONFIG in "${MASK_CONFIGS[@]}"; do
        IFS='|' read -r MASK_TAG MASK_RATIO NUM_PATCHES <<< "$CONFIG"
        MASKED_DIR="outputs/masked_frames/${CATEGORY}/${SEQ_ID}/${MASK_TAG}"

        if [ -d "$MASKED_DIR" ] && [ "$(ls -A $MASKED_DIR 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [skip] ${CATEGORY} ${MASK_TAG}"
        else
            echo "  [gen]  ${CATEGORY} ${MASK_TAG} ..."
            python scripts/masking_exp/mask.py \
                --images_dir  "$IMAGES_DIR" \
                --output_dir  "$MASKED_DIR" \
                --mask_ratio  "$MASK_RATIO" \
                --num_patches "$NUM_PATCHES" \
                --seed        42 &
            MASK_PIDS+=($!)
        fi
    done
done

for PID in "${MASK_PIDS[@]}"; do wait "$PID"; done
echo "  All masked frames ready."

# ── Step 2: Build work queue ───────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Building work queue ..."

WORK_QUEUE=()
for N in $(seq $N_FRAMES_MIN $N_FRAMES_MAX); do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"

        for CONFIG in "${MASK_CONFIGS[@]}"; do
            IFS='|' read -r MASK_TAG MASK_RATIO NUM_PATCHES <<< "$CONFIG"
            MASKED_DIR="outputs/masked_frames/${CATEGORY}/${SEQ_ID}/${MASK_TAG}"
            OUT="outputs/dust3r/mask_multi/${MASK_TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"

            if [ ! -f "${OUT}/metrics.txt" ]; then
                WORK_QUEUE+=("${N}|${SEQ_DIR}|${MASKED_DIR}|${MASK_RATIO}|${OUT}")
            fi
        done
    done
done

TOTAL=${#WORK_QUEUE[@]}
echo "  Total tasks: ${TOTAL}  (2 mask ratios × 6 categories × up to 39 n_frames)"

# ── Step 3: Dispatch across 4 GPUs ────────────────────────────────────────────
echo ""
echo ">>> Step 3: Dispatching DUSt3R inference ..."

declare -a GPU_PID
for (( g=0; g<N_GPUS; g++ )); do GPU_PID[$g]=0; done

COMPLETED=0
JOB_IDX=0

while [ $JOB_IDX -lt $TOTAL ] || [ $COMPLETED -lt $TOTAL ]; do
    for (( g=0; g<N_GPUS; g++ )); do
        PID=${GPU_PID[$g]}

        if [ $PID -ne 0 ]; then
            if ! kill -0 $PID 2>/dev/null; then
                wait $PID
                STATUS=$?
                COMPLETED=$((COMPLETED + 1))
                [ $STATUS -eq 0 ] \
                    && echo "  [OK  ${COMPLETED}/${TOTAL}] GPU ${g}" \
                    || echo "  [FAIL ${COMPLETED}/${TOTAL}] GPU ${g} exit=${STATUS}"
                GPU_PID[$g]=0
                PID=0
            fi
        fi

        if [ $PID -eq 0 ] && [ $JOB_IDX -lt $TOTAL ]; then
            IFS='|' read -r N SEQ_DIR MASKED_DIR MASK_RATIO OUT <<< "${WORK_QUEUE[$JOB_IDX]}"
            mkdir -p "$OUT"

            PYTORCH_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES=$g \
            python scripts/run_dust3r_inference.py \
                --sequence_dir "$SEQ_DIR" \
                --dust3r_dir   "dust3r" \
                --n_frames     "$N" \
                --n_masked     "$N" \
                --masked_dir   "$MASKED_DIR" \
                --mask_ratio   "$MASK_RATIO" \
                --output_dir   "$OUT" \
                > "${OUT}/inference.log" 2>&1 &

            GPU_PID[$g]=$!
            MASK_TAG=$(basename $(dirname $(dirname $MASKED_DIR)))_$(basename $(dirname $MASKED_DIR))
            echo "  GPU ${g} → n=${N}  $(basename $SEQ_DIR)  mask=$(basename $MASKED_DIR)  [${JOB_IDX+1}/${TOTAL}]"
            JOB_IDX=$((JOB_IDX + 1))
        fi
    done
    sleep 5
done

echo ""
echo "  All ${COMPLETED} inference runs complete."

# ── Step 4: Generate plot ──────────────────────────────────────────────────────
echo ""
echo ">>> Step 4: Generating plot ..."
python scripts/masking_exp/plot_masking_multi.py \
    --results_root "outputs/dust3r/mask_multi" \
    --n_min "$N_FRAMES_MIN" \
    --n_max "$N_FRAMES_MAX"

echo ""
echo "================================================================"
echo "Job finished : $(date)"
echo "Plot saved to: outputs/dust3r/mask_multi/masking_compare.png"
echo "================================================================"
