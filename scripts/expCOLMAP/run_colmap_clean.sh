#!/bin/bash
#SBATCH --job-name=colmap_clean
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/colmap_clean_%j.log
#SBATCH --error=logs/colmap_clean_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

FRAMES_TO_TRY=(50 60 70 80 90 100 110 120 130 140 150)
N_GPUS=2
SEQ_BASE="data/co3d"
OUTPUT_ROOT="outputs/colmap/clean"

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

mkdir -p logs

# ── Build work queue (no masking step needed — use clean images directly) ─────
echo ""
echo ">>> Building work queue (clean frames) ..."

WORK_QUEUE=()
for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    IMAGES_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}/images"
    MAX_AVAIL=$(ls "$IMAGES_DIR" 2>/dev/null | grep -Ec '\.(jpg|jpeg|png)$')

    for N in "${FRAMES_TO_TRY[@]}"; do
        if [ "$N" -gt "$MAX_AVAIL" ]; then
            echo "  [skip] ${CATEGORY} n=${N} (only ${MAX_AVAIL} frames available)"
            continue
        fi
        OUT_DIR="${OUTPUT_ROOT}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%03d' $N)"
        if grep -q "status:.*ok" "${OUT_DIR}/metrics.txt" 2>/dev/null; then
            echo "  [done] ${CATEGORY} n=${N}"
            continue
        fi
        WORK_QUEUE+=("${CATEGORY}|${SEQ_ID}|${N}")
    done
done

TOTAL=${#WORK_QUEUE[@]}
echo ">>> Total tasks: ${TOTAL}  (across ${N_GPUS} GPUs)"

# ── Dispatch across GPUs ───────────────────────────────────────────────────────
echo ""
echo ">>> Dispatching tasks ..."

declare -a GPU_PID
for (( i=0; i<N_GPUS; i++ )); do GPU_PID[$i]=0; done

TASK_IDX=0
DONE=0

while [ $DONE -lt $TOTAL ]; do
    for (( g=0; g<N_GPUS; g++ )); do
        PID=${GPU_PID[$g]}

        if [ "$PID" -ne 0 ] && kill -0 "$PID" 2>/dev/null; then
            continue
        fi
        GPU_PID[$g]=0

        if [ $TASK_IDX -lt $TOTAL ]; then
            IFS='|' read -r CATEGORY SEQ_ID N <<< "${WORK_QUEUE[$TASK_IDX]}"

            SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"
            IMAGES_DIR="${SEQ_DIR}/images"
            OUT_DIR="${OUTPUT_ROOT}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%03d' $N)"

            echo "  [GPU ${g}] task $((TASK_IDX+1))/${TOTAL}: ${CATEGORY} n=${N}"
            mkdir -p "$OUT_DIR"

            python scripts/expCOLMAP/run_colmap_inference.py \
                --masked_dir   "$IMAGES_DIR" \
                --sequence_dir "$SEQ_DIR" \
                --output_dir   "$OUT_DIR" \
                --n_frames     "$N" \
                --gpu_index    "$g" \
                > "${OUT_DIR}/run.log" 2>&1 &

            GPU_PID[$g]=$!
            TASK_IDX=$((TASK_IDX + 1))
        fi
    done

    DONE=0
    running=0
    for (( i=0; i<N_GPUS; i++ )); do
        if [ "${GPU_PID[$i]}" -ne 0 ] && kill -0 "${GPU_PID[$i]}" 2>/dev/null; then
            running=$((running + 1))
        else
            GPU_PID[$i]=0
        fi
    done
    DONE=$(( TASK_IDX - running ))

    if [ $TASK_IDX -lt $TOTAL ] || [ $running -gt 0 ]; then
        sleep 5
    fi
done

for (( i=0; i<N_GPUS; i++ )); do
    [ "${GPU_PID[$i]}" -ne 0 ] && wait "${GPU_PID[$i]}"
done

echo ""
echo ">>> All ${TOTAL} tasks complete."

# ── Plot ───────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Generating plot ..."
python scripts/expCOLMAP/plot_colmap_clean.py \
    --results_root "$OUTPUT_ROOT" \
    --output_dir   "$OUTPUT_ROOT"

echo ""
echo "================================================================"
echo "Job finished : $(date)"
echo "Plot         : ${OUTPUT_ROOT}/colmap_clean_chamfer_vs_nframes.png"
echo "================================================================"
