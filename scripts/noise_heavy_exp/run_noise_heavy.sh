#!/bin/bash
#SBATCH --job-name=noise_heavy
#SBATCH --partition=h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/noise_heavy_%j.log
#SBATCH --error=logs/noise_heavy_%j.err

# ── Configuration ──────────────────────────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

NOISE_STDS=(200 300)
N_FRAMES=20
N_GPUS=4
SEQ_BASE="data/co3d"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan

echo "================================================================"
echo "Job started  : $(date)"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Sequences    : ${!SEQUENCES[*]}"
echo "Noise stds   : ${NOISE_STDS[*]}"
echo "n_frames     : ${N_FRAMES}"
echo "GPUs         : ${N_GPUS}"
echo "================================================================"

mkdir -p logs

# ── Step 1: Generate noisy frames (CPU, parallel) ──────────────────────────────
echo ""
echo ">>> Step 1: Generating degraded frames (noise std=200, 300) ..."

DEGRADE_PIDS=()
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    IMAGES_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images"

    for STD in "${NOISE_STDS[@]}"; do
        OUT="outputs/degraded_frames/${CATEGORY}/${SEQ_ID}/noise_s${STD}"
        if [ -d "$OUT" ] && [ "$(ls -A $OUT 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [skip] ${CATEGORY} noise_s${STD} (already exists)"
        else
            python3 scripts/gaussian_noise_and_blur_exps/degrade.py \
                --images_dir "$IMAGES_DIR" \
                --output_dir "$OUT" \
                --mode noise \
                --noise_std "$STD" \
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
echo ">>> Step 2: Building work queue (n_frames=${N_FRAMES}) ..."

WORK_QUEUE=()
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"

    for STD in "${NOISE_STDS[@]}"; do
        TAG="noise_s${STD}"
        OUT="outputs/dust3r/noise_heavy/${TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N_FRAMES)"
        if [ ! -f "${OUT}/metrics.txt" ]; then
            DEGRADED="outputs/degraded_frames/${CATEGORY}/${SEQ_ID}/${TAG}"
            WORK_QUEUE+=("${SEQ_DIR}|${DEGRADED}|${OUT}")
        else
            echo "  [skip] ${CATEGORY} ${TAG} n=${N_FRAMES} (metrics.txt exists)"
        fi
    done
done

TOTAL_JOBS=${#WORK_QUEUE[@]}
echo "  Total runs in queue: ${TOTAL_JOBS}"

# ── Step 3: Dispatch across GPUs ──────────────────────────────────────────────
echo ""
echo ">>> Step 3: Dispatching DUSt3R inference across ${N_GPUS} GPUs ..."

declare -a GPU_PID
for (( g=0; g<N_GPUS; g++ )); do GPU_PID[$g]=0; done

COMPLETED=0
JOB_IDX=0

while [ $JOB_IDX -lt $TOTAL_JOBS ] || [ $COMPLETED -lt $TOTAL_JOBS ]; do
    for (( g=0; g<N_GPUS; g++ )); do
        PID=${GPU_PID[$g]}

        if [ $PID -ne 0 ]; then
            if ! kill -0 $PID 2>/dev/null; then
                wait $PID
                STATUS=$?
                COMPLETED=$((COMPLETED + 1))
                [ $STATUS -eq 0 ] \
                    && echo "    [OK ${COMPLETED}/${TOTAL_JOBS}] GPU $g done" \
                    || echo "    [FAIL] GPU $g exit=$STATUS"
                GPU_PID[$g]=0
                PID=0
            fi
        fi

        if [ $PID -eq 0 ] && [ $JOB_IDX -lt $TOTAL_JOBS ]; then
            IFS='|' read -r SEQ_DIR DEGRADED_DIR OUT_SUBDIR <<< "${WORK_QUEUE[$JOB_IDX]}"
            mkdir -p "$OUT_SUBDIR"

            PYTORCH_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES=$g \
            python3 scripts/run_dust3r_inference.py \
                --sequence_dir "$SEQ_DIR" \
                --dust3r_dir   "dust3r" \
                --n_frames     "$N_FRAMES" \
                --n_masked     "$N_FRAMES" \
                --masked_dir   "$DEGRADED_DIR" \
                --output_dir   "$OUT_SUBDIR" \
                --mask_ratio   0.0 \
                > "${OUT_SUBDIR}/inference.log" 2>&1 &

            GPU_PID[$g]=$!
            echo "  GPU $g → $(basename $SEQ_DIR)  $(basename $DEGRADED_DIR)  [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
            JOB_IDX=$((JOB_IDX + 1))
        fi
    done
    sleep 5
done

echo ""
echo "  All ${COMPLETED} inference runs complete."

# ── Step 4: Plot ───────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 4: Generating plots ..."
python3 scripts/noise_heavy_exp/plot_noise_heavy.py \
    --results_root "outputs/dust3r/noise_heavy" \
    --n_frames "$N_FRAMES"

echo ""
echo "================================================================"
echo "Job finished : $(date)"
echo "Plots saved to: outputs/dust3r/noise_heavy/"
echo "================================================================"
