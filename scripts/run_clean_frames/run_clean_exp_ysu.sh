#!/bin/bash
#SBATCH --job-name=clean_2_20
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/clean_exp_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/clean_exp_%j.err

# DUSt3R clean-frame baseline
# n_frames: 2..20, all 6 CO3D categories
# 19 n_frames x 6 categories = 114 runs

declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

N_FRAMES_MIN=2
N_FRAMES_MAX=20
N_GPUS=4
PYTHON="/mnt/weka/hminasyan/co3d_env/bin/python3"
PROJECT_DIR="/home/hminasyan/project_Hayk_Minasyan"
SEQ_BASE="/mnt/weka/hminasyan/data/co3d"
OUTPUT_ROOT="/mnt/weka/hminasyan/outputs/dust3r/clean_frames"

cd "$PROJECT_DIR"
mkdir -p logs

echo "================================================================"
echo "Started  : $(date)"
echo "Node     : $(hostname)"
echo "Mode     : clean frames only"
echo "n_frames : ${N_FRAMES_MIN} → ${N_FRAMES_MAX}"
echo "GPUs     : ${N_GPUS}"
echo "================================================================"

# ── Step 1: Build work queue ───────────────────────────────────────────────────
echo ""
echo ">>> Step 1: Building work queue ..."

WORK_QUEUE=()
for N in $(seq $N_FRAMES_MIN $N_FRAMES_MAX); do
    for CATEGORY in "${!SEQUENCES[@]}"; do
        SEQ_ID="${SEQUENCES[$CATEGORY]}"
        SEQ_DIR="${SEQ_BASE}/${CATEGORY}/${SEQ_ID}"
        OUT="${OUTPUT_ROOT}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"

        if [ ! -f "${OUT}/metrics.txt" ]; then
            WORK_QUEUE+=("${N}|${SEQ_DIR}|${OUT}")
        else
            echo "  [skip] ${CATEGORY} n=${N}"
        fi
    done
done

TOTAL_JOBS=${#WORK_QUEUE[@]}
echo "  Total runs: ${TOTAL_JOBS}"

# ── Step 2: Dispatch across GPUs ──────────────────────────────────────────────
echo ""
echo ">>> Step 2: Running DUSt3R inference on ${N_GPUS} GPUs ..."

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
            IFS='|' read -r N SEQ_DIR OUT_SUBDIR <<< "${WORK_QUEUE[$JOB_IDX]}"
            mkdir -p "$OUT_SUBDIR"

            PYTORCH_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES=$g \
            $PYTHON scripts/run_dust3r_inference.py \
                --sequence_dir "$SEQ_DIR" \
                --dust3r_dir   "dust3r" \
                --n_frames     "$N" \
                --output_dir   "$OUT_SUBDIR" \
                > "${OUT_SUBDIR}/inference.log" 2>&1 &

            GPU_PID[$g]=$!
            echo "  GPU $g → n=${N}  $(basename "$SEQ_DIR")  clean  [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
            JOB_IDX=$((JOB_IDX + 1))
        fi
    done
    sleep 5
done

echo ""
echo "  All ${COMPLETED} inference runs complete."
echo "================================================================"
echo "Finished : $(date)"
echo "Results  : ${OUTPUT_ROOT}/"
echo "Transfer : rsync -av hminasyan@cluster.ysu.am:${OUTPUT_ROOT}/ outputs/dust3r/clean_frames/"
echo "================================================================"
