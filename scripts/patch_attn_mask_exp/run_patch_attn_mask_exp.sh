#!/bin/bash
#SBATCH --job-name=patch_attn_mask
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/patch_attn_mask_exp_%j.log
#SBATCH --error=logs/patch_attn_mask_exp_%j.err

# Random patch-key masking (DUSt3R-aligned grid): pixels stay clean; transformer
# cannot attend to masked patch *keys* (see dust3r/utils/patch_attn_mask.py).

declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

MASK_RATIOS=(0.05 0.10 0.25 0.50)
MASK_TAGS=(mask_5pct mask_10pct mask_25pct mask_50pct)

N_FRAMES_MIN=2
N_FRAMES_MAX=20
N_GPUS=4
SEQ_BASE="data/co3d"
MASK_ROOT="outputs/patch_attn_masks"
OUTPUT_ROOT="outputs/dust3r/patch_attn_mask_exp"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started  : $(date)"
echo "Node         : $(hostname)"
echo "Mask npy dir : ${MASK_ROOT}/<cat>/<seq>/<tag>/"
echo "Results      : ${OUTPUT_ROOT}/<tag>/..."
echo "================================================================"

mkdir -p logs

echo ""
echo ">>> Generating patch masks (skip if folder exists and non-empty) ..."
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    IMG_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images"
    for i in "${!MASK_RATIOS[@]}"; do
        TAG="${MASK_TAGS[$i]}"
        RATIO="${MASK_RATIOS[$i]}"
        OUT_MASK="${MASK_ROOT}/${CATEGORY}/${SEQ_ID}/${TAG}"
        if [ -d "$OUT_MASK" ] && [ "$(find "$OUT_MASK" -maxdepth 1 -name '*.patch_mask_hw.npy' 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [skip masks] ${CATEGORY}/${SEQ_ID} ${TAG}"
        else
            mkdir -p "$OUT_MASK"
            python3 scripts/patch_attn_mask_exp/generate_patch_attn_masks.py \
                --images_dir "$IMG_DIR" \
                --output_dir "$OUT_MASK" \
                --mask_ratio "$RATIO" \
                --seed 42
        fi
    done
done

echo ""
echo ">>> Building inference queue ..."
WORK_QUEUE=()
for N in $(seq $N_FRAMES_MIN $N_FRAMES_MAX); do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"
        for i in "${!MASK_RATIOS[@]}"; do
            TAG="${MASK_TAGS[$i]}"
            NPY_DIR="${MASK_ROOT}/${CATEGORY}/${SEQ_ID}/${TAG}"
            OUT="${OUTPUT_ROOT}/${TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"
            if [ ! -f "${OUT}/metrics.txt" ]; then
                WORK_QUEUE+=("${N}|${SEQ_DIR}|${NPY_DIR}|${OUT}|${MASK_RATIOS[$i]}")
            fi
        done
    done
done

TOTAL_JOBS=${#WORK_QUEUE[@]}
echo "  Total runs: ${TOTAL_JOBS}"

echo ""
echo ">>> DUSt3R inference (${N_GPUS} GPUs) ..."
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
                    && echo "    [OK ${COMPLETED}/${TOTAL_JOBS}] GPU $g" \
                    || echo "    [FAIL] GPU $g exit=$STATUS"
                GPU_PID[$g]=0
                PID=0
            fi
        fi
        if [ $PID -eq 0 ] && [ $JOB_IDX -lt $TOTAL_JOBS ]; then
            IFS='|' read -r N SEQ_DIR NPY_DIR OUT_SUBDIR RATIO <<< "${WORK_QUEUE[$JOB_IDX]}"
            mkdir -p "$OUT_SUBDIR"
            PYTORCH_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES=$g \
            python3 scripts/run_dust3r_inference.py \
                --sequence_dir "$SEQ_DIR" \
                --dust3r_dir   "dust3r" \
                --n_frames     "$N" \
                --n_masked     0 \
                --output_dir   "$OUT_SUBDIR" \
                --mask_ratio   "$RATIO" \
                --attn_mask_npy_dir "$NPY_DIR" \
                > "${OUT_SUBDIR}/inference.log" 2>&1 &
            GPU_PID[$g]=$!
            echo "  GPU $g → n=${N} $(basename "$SEQ_DIR") $(basename "$NPY_DIR") [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
            JOB_IDX=$((JOB_IDX + 1))
        fi
    done
    sleep 5
done

echo ""
echo "  All ${COMPLETED} inference runs complete."

echo ""
echo ">>> Plot ..."
python3 scripts/patch_attn_mask_exp/plot_patch_attn_mask_exp.py \
    --results_root "$OUTPUT_ROOT" \
    --n_min "$N_FRAMES_MIN" \
    --n_max "$N_FRAMES_MAX"

echo ""
echo "================================================================"
echo "Finished : $(date)"
echo "Plot       : ${OUTPUT_ROOT}/patch_attn_mask_compare.png"
echo "================================================================"
