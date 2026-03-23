#!/bin/bash
#SBATCH --job-name=grid_new_ratios
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grid_new_ratios_%j.log
#SBATCH --error=logs/grid_new_ratios_%j.err

# New mask ratios: 20%, 30%, 40%
# n_frames: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
# 6 categories x 3 ratios x 10 n_frames = 180 runs

declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

MASK_RATIOS=(0.20 0.30 0.40)
MASK_TAGS=(mask_20pct mask_30pct mask_40pct)
N_FRAMES_LIST=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
N_GPUS=4
SEQ_BASE="/mnt/weka/hminasyan/data/co3d"
OUTPUT_ROOT="/mnt/weka/hminasyan/outputs/dust3r/grid_mask_exp"
GRID_MASKED_ROOT="/mnt/weka/hminasyan/outputs/grid_masked"
PROJECT_DIR="/home/hminasyan/project_Hayk_Minasyan"

PYTHON="/mnt/weka/hminasyan/co3d_env/bin/python3"
cd $PROJECT_DIR

echo "================================================================"
echo "Started  : $(date)"
echo "Node     : $(hostname)"
echo "Ratios   : ${MASK_TAGS[*]}"
echo "n_frames : ${N_FRAMES_LIST[*]}"
echo "GPUs     : ${N_GPUS}"
echo "================================================================"

mkdir -p logs

# ── Step 1: Generate grid-masked frames for new ratios ────────────────────────
echo ""
echo ">>> Step 1: Generating grid-masked frames (20%, 30%, 40%) ..."

for i in "${!MASK_RATIOS[@]}"; do
    RATIO="${MASK_RATIOS[$i]}"
    TAG="${MASK_TAGS[$i]}"
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        OUT_DIR="${GRID_MASKED_ROOT}/${CATEGORY}/${SEQ_ID}/${TAG}"
        if [ -d "$OUT_DIR" ] && [ "$(ls -A "$OUT_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [skip] ${CATEGORY} ${TAG}"
        else
            echo "  [gen]  ${CATEGORY} ${TAG} ..."
            $PYTHON scripts/masking_exp/grid_mask.py \
                --images_dir "${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images" \
                --output_dir "$OUT_DIR" \
                --patch_size 16 \
                --mask_ratio "$RATIO" \
                --seed 42
        fi
    done
done
echo "  All masked frames ready."

# ── Step 2: Build work queue ───────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Building work queue ..."

WORK_QUEUE=()
for N in "${N_FRAMES_LIST[@]}"; do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"
        for i in "${!MASK_TAGS[@]}"; do
            TAG="${MASK_TAGS[$i]}"
            MASKED_DIR="${GRID_MASKED_ROOT}/${CATEGORY}/${SEQ_ID}/${TAG}"
            OUT="${OUTPUT_ROOT}/${TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"
            if [ ! -f "${OUT}/metrics.txt" ]; then
                WORK_QUEUE+=("${N}|${SEQ_DIR}|${MASKED_DIR}|${OUT}")
            else
                echo "  [skip] ${CATEGORY} ${TAG} n=${N}"
            fi
        done
    done
done

TOTAL_JOBS=${#WORK_QUEUE[@]}
echo "  Total runs: ${TOTAL_JOBS}"

# ── Step 3: Dispatch across GPUs ──────────────────────────────────────────────
echo ""
echo ">>> Step 3: Running DUSt3R inference on ${N_GPUS} GPUs ..."

declare -a GPU_PID
for (( g=0; g<N_GPUS; g++ )); do GPU_PID[$g]=0; done

COMPLETED=0
JOB_IDX=0

while [ $JOB_IDX -lt $TOTAL_JOBS ] || [ $COMPLETED -lt $TOTAL_JOBS ]; do
    for (( g=0; g<N_GPUS; g++ )); do
        PID=${GPU_PID[$g]}

        if [ $PID -ne 0 ]; then
            if ! kill -0 $PID 2>/dev/null; then
                wait $PID; STATUS=$?
                COMPLETED=$((COMPLETED + 1))
                [ $STATUS -eq 0 ] \
                    && echo "    [OK ${COMPLETED}/${TOTAL_JOBS}] GPU $g done" \
                    || echo "    [FAIL] GPU $g exit=$STATUS"
                GPU_PID[$g]=0
                PID=0
            fi
        fi

        if [ $PID -eq 0 ] && [ $JOB_IDX -lt $TOTAL_JOBS ]; then
            IFS='|' read -r N SEQ_DIR MASKED_DIR OUT_SUBDIR <<< "${WORK_QUEUE[$JOB_IDX]}"
            mkdir -p "$OUT_SUBDIR"

            PYTORCH_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES=$g \
            $PYTHON scripts/run_dust3r_inference.py \
                --sequence_dir "$SEQ_DIR" \
                --dust3r_dir   "dust3r" \
                --n_frames     "$N" \
                --n_masked     "$N" \
                --masked_dir   "$MASKED_DIR" \
                --output_dir   "$OUT_SUBDIR" \
                --mask_ratio   0.0 \
                > "${OUT_SUBDIR}/inference.log" 2>&1 &

            GPU_PID[$g]=$!
            TAG_NAME=$(basename "$MASKED_DIR")
            echo "  GPU $g → n=${N}  $(basename $SEQ_DIR)  ${TAG_NAME}  [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
            JOB_IDX=$((JOB_IDX + 1))
        fi
    done
    sleep 5
done

echo ""
echo "  All ${COMPLETED} inference runs complete."

# ── Step 4: Plot (all 5 ratios together) ──────────────────────────────────────
echo ""
echo ">>> Step 4: Generating updated plot ..."
$PYTHON scripts/grid_mask_exp/plot_grid_mask_exp.py \
    --results_root "$OUTPUT_ROOT" \
    --n_min 2 --n_max 20

echo ""
echo "================================================================"
echo "Finished : $(date)"
echo "Plot     : ${OUTPUT_ROOT}/grid_mask_compare.png"
echo "================================================================"
