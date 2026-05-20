#!/bin/bash
# Run noise heavy experiment on the LOGIN NODE using GPUs 0-3.
# n_frames: 10, 14, 16, 18, 20
# noise stds: 200, 300
# 6 CO3D categories — averaged Chamfer Distance plotted at the end.
#
# Usage:
#   bash scripts/noise_heavy_exp/run_noise_heavy_local.sh

set -e

# ── Configuration ──────────────────────────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

NOISE_STDS=(200 300)
N_FRAMES_LIST=(10 14 16 18 20)
N_GPUS=4
SEQ_BASE="data/co3d"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Started  : $(date)"
echo "Sequences: ${!SEQUENCES[*]}"
echo "Noise    : ${NOISE_STDS[*]}"
echo "n_frames : ${N_FRAMES_LIST[*]}"
echo "GPUs     : 0-$((N_GPUS-1))"
echo "================================================================"

mkdir -p logs

# ── Step 1: Generate noisy frames (CPU, parallel) ──────────────────────────────
echo ""
echo ">>> Step 1: Generating degraded frames ..."

DEGRADE_PIDS=()
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    IMAGES_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images"

    for STD in "${NOISE_STDS[@]}"; do
        OUT="outputs/degraded_frames/${CATEGORY}/${SEQ_ID}/noise_s${STD}"
        if [ -d "$OUT" ] && [ "$(ls -A "$OUT" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [skip] ${CATEGORY} noise_s${STD}"
        else
            python3 scripts/gaussian_noise_and_blur_exps/degrade.py \
                --images_dir "$IMAGES_DIR" \
                --output_dir "$OUT" \
                --mode noise \
                --noise_std  "$STD" \
                --seed 42 &
            DEGRADE_PIDS+=($!)
            echo "  [gen]  ${CATEGORY} noise_s${STD}"
        fi
    done
done

for pid in "${DEGRADE_PIDS[@]}"; do wait "$pid"; done
echo "  All degraded frames ready."

# ── Step 2: Build work queue ───────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Building work queue ..."

WORK_QUEUE=()
for N in "${N_FRAMES_LIST[@]}"; do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"

        for STD in "${NOISE_STDS[@]}"; do
            TAG="noise_s${STD}"
            OUT="outputs/dust3r/noise_heavy/${TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"
            if [ ! -f "${OUT}/metrics.txt" ]; then
                DEGRADED="outputs/degraded_frames/${CATEGORY}/${SEQ_ID}/${TAG}"
                WORK_QUEUE+=("${N}|${SEQ_DIR}|${DEGRADED}|${OUT}")
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
            IFS='|' read -r N SEQ_DIR DEGRADED_DIR OUT_SUBDIR <<< "${WORK_QUEUE[$JOB_IDX]}"
            mkdir -p "$OUT_SUBDIR"

            PYTORCH_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES=$g \
            python3 scripts/run_dust3r_inference.py \
                --sequence_dir "$SEQ_DIR" \
                --dust3r_dir   "dust3r" \
                --n_frames     "$N" \
                --n_masked     "$N" \
                --masked_dir   "$DEGRADED_DIR" \
                --output_dir   "$OUT_SUBDIR" \
                --mask_ratio   0.0 \
                > "${OUT_SUBDIR}/inference.log" 2>&1 &

            GPU_PID[$g]=$!
            echo "  GPU $g → n=${N}  $(basename $SEQ_DIR)  noise_s$(basename $DEGRADED_DIR | sed 's/noise_s//')  [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
            JOB_IDX=$((JOB_IDX + 1))
        fi
    done
    sleep 5
done

echo ""
echo "  All ${COMPLETED} inference runs complete."

# ── Step 4: Plot ───────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 4: Generating plot ..."
python3 scripts/noise_heavy_exp/plot_noise_heavy.py \
    --results_root "outputs/dust3r/noise_heavy"

echo ""
echo "================================================================"
echo "Finished : $(date)"
echo "Plot     : outputs/dust3r/noise_heavy/noise_heavy_compare.png"
echo "================================================================"
