#!/bin/bash
#SBATCH --job-name=noise_exp
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/noise_exp_%j.log
#SBATCH --error=logs/noise_exp_%j.err

# DUSt3R inference for Gaussian noise (default σ=5,10,30,50,70).
# n_frames: 2..20, all 6 CO3D categories — skips runs that already have metrics.txt
#
# Parallel σ=10-only job (same 4 GPUs, separate queue slot):
#   sbatch scripts/noise_exp_30_50_70/run_noise_s10_parallel.sh
# Or any subset:
#   sbatch --export=ALL,NOISE_STDS_EXPORT="10 20" scripts/noise_exp_30_50_70/run_noise_exp.sh

declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

if [ -n "${NOISE_STDS_EXPORT:-}" ]; then
    read -ra NOISE_STDS <<< "$NOISE_STDS_EXPORT"
else
    NOISE_STDS=(5 10 30 50 70)
fi
N_FRAMES_MIN=2
N_FRAMES_MAX=20
N_GPUS=4
SEQ_BASE="data/co3d"
OUTPUT_ROOT="outputs/dust3r/noise_exp_30_50_70"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Started  : $(date)"
echo "Node     : $(hostname)"
echo "Noise    : ${NOISE_STDS[*]}"
echo "n_frames : ${N_FRAMES_MIN} → ${N_FRAMES_MAX}"
echo "GPUs     : ${N_GPUS}"
echo "================================================================"

mkdir -p logs

# ── Step 1: Generate missing noise frames ─────────────────────────────────────
echo ""
echo ">>> Step 1: Generating missing noise frames ..."

for STD in "${NOISE_STDS[@]}"; do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        OUT="outputs/degraded_frames/${CATEGORY}/${SEQ_ID}/noise_s${STD}"
        if [ -d "$OUT" ] && [ "$(ls -A "$OUT" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [skip] ${CATEGORY} noise_s${STD}"
        else
            echo "  [gen]  ${CATEGORY} noise_s${STD} ..."
            python3 scripts/gaussian_noise_and_blur_exps/degrade.py \
                --images_dir "${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images" \
                --output_dir "$OUT" \
                --mode noise \
                --noise_std "$STD" \
                --seed 42
        fi
    done
done
echo "  All noise frames ready."

# ── Step 2: Build work queue ───────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Building work queue ..."

WORK_QUEUE=()
for N in $(seq $N_FRAMES_MIN $N_FRAMES_MAX); do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"
        for STD in "${NOISE_STDS[@]}"; do
            TAG="noise_s${STD}"
            OUT="${OUTPUT_ROOT}/${TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"
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
            STD_TAG=$(basename "$DEGRADED_DIR")
            echo "  GPU $g → n=${N}  $(basename $SEQ_DIR)  ${STD_TAG}  [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
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
python3 scripts/noise_exp_30_50_70/plot_noise_exp.py \
    --results_root "$OUTPUT_ROOT"

echo ""
echo "================================================================"
echo "Finished : $(date)"
echo "Plot     : ${OUTPUT_ROOT}/noise_exp_compare.png"
echo "================================================================"
