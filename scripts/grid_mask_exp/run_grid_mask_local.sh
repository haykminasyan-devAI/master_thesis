#!/bin/bash
# Run grid masking DUSt3R experiment on the LOGIN NODE using GPUs 0-3.
# 6 categories x 3 mask ratios (10%, 25%, 50%) x n_frames=2..20 = 342 runs.
#
# Usage:
#   bash scripts/grid_mask_exp/run_grid_mask_local.sh

set -e

# ── Configuration ──────────────────────────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

MASK_TAGS=(mask_10pct mask_25pct mask_50pct)

N_FRAMES_MIN=2
N_FRAMES_MAX=20
N_GPUS=4
SEQ_BASE="data/co3d"
OUTPUT_ROOT="outputs/dust3r/grid_mask_exp"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Started  : $(date)"
echo "Node     : $(hostname)"
echo "Sequences: ${!SEQUENCES[*]}"
echo "n_frames : ${N_FRAMES_MIN} → ${N_FRAMES_MAX}"
echo "GPUs     : 0-$((N_GPUS-1))"
echo "================================================================"

mkdir -p logs

# ── Build work queue ───────────────────────────────────────────────────────────
echo ""
echo ">>> Building work queue ..."

WORK_QUEUE=()
for N in $(seq $N_FRAMES_MIN $N_FRAMES_MAX); do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"

        for TAG in "${MASK_TAGS[@]}"; do
            MASKED_DIR="outputs/grid_masked/${CATEGORY}/${SEQ_ID}/${TAG}"
            OUT="${OUTPUT_ROOT}/${TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"

            if [ ! -f "${OUT}/metrics.txt" ]; then
                WORK_QUEUE+=("${N}|${SEQ_DIR}|${MASKED_DIR}|${OUT}")
            fi
        done
    done
done

TOTAL_JOBS=${#WORK_QUEUE[@]}
echo "  Total runs: ${TOTAL_JOBS}"

# ── Dispatch across GPUs ───────────────────────────────────────────────────────
echo ""
echo ">>> Dispatching DUSt3R inference across ${N_GPUS} GPUs ..."

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
            python3 scripts/run_dust3r_inference.py \
                --sequence_dir "$SEQ_DIR" \
                --dust3r_dir   "dust3r" \
                --n_frames     "$N" \
                --n_masked     "$N" \
                --masked_dir   "$MASKED_DIR" \
                --output_dir   "$OUT_SUBDIR" \
                --mask_ratio   0.0 \
                > "${OUT_SUBDIR}/inference.log" 2>&1 &

            GPU_PID[$g]=$!
            echo "  GPU $g → n=${N}  $(basename $SEQ_DIR)  ${TAG}  [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
            JOB_IDX=$((JOB_IDX + 1))
        fi
    done
    sleep 5
done

echo ""
echo "  All ${COMPLETED} inference runs complete."

# ── Plot ───────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Generating plot ..."
python3 scripts/grid_mask_exp/plot_grid_mask_exp.py \
    --results_root "$OUTPUT_ROOT" \
    --n_min "$N_FRAMES_MIN" \
    --n_max "$N_FRAMES_MAX"

echo ""
echo "================================================================"
echo "Finished : $(date)"
echo "Plot     : ${OUTPUT_ROOT}/grid_mask_compare.png"
echo "================================================================"
