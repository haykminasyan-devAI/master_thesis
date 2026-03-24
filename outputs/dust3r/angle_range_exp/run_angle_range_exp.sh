#!/bin/bash
#SBATCH --job-name=angle_range
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/angle_range_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/angle_range_%j.err

# Angle range experiment: how does limiting the viewpoint coverage affect DUSt3R?
# 3 angle ranges (0-60°, 0-90°, 0-180°) x 6 sequences x n_frames=2..10

set -e

# ── Configuration ──────────────────────────────────────────────────────────────
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

# angle range tag → first X% of frames
declare -A ANGLE_RANGES
ANGLE_RANGES["range_0_60deg"]=16.67    # 60/360 * 100
ANGLE_RANGES["range_0_90deg"]=25.0     # 90/360 * 100
ANGLE_RANGES["range_0_180deg"]=50.0    # 180/360 * 100

N_FRAMES_MIN=2
N_FRAMES_MAX=10
N_GPUS=4
PYTHON="/mnt/weka/hminasyan/co3d_env/bin/python3"
PROJECT_DIR="/home/hminasyan/project_Hayk_Minasyan"
SEQ_BASE="/mnt/weka/hminasyan/data/co3d"
OUTPUT_ROOT="/mnt/weka/hminasyan/outputs/dust3r/angle_range_exp"

cd $PROJECT_DIR
mkdir -p logs

echo "================================================================"
echo "Started  : $(date)"
echo "Node     : $(hostname)"
echo "Sequences: ${!SEQUENCES[*]}"
echo "n_frames : ${N_FRAMES_MIN} → ${N_FRAMES_MAX}"
echo "Ranges   : ${!ANGLE_RANGES[*]}"
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

        for TAG in "${!ANGLE_RANGES[@]}"; do
            PCT="${ANGLE_RANGES[$TAG]}"
            OUT="${OUTPUT_ROOT}/${TAG}/${CATEGORY}_${SEQ_ID}/frames_$(printf '%02d' $N)"

            if [ ! -f "${OUT}/metrics.txt" ]; then
                WORK_QUEUE+=("${N}|${SEQ_DIR}|${PCT}|${OUT}")
            fi
        done
    done
done

TOTAL_JOBS=${#WORK_QUEUE[@]}
echo "  Total runs: ${TOTAL_JOBS}"

# ── Dispatch across GPUs ───────────────────────────────────────────────────────
echo ""
echo ">>> Dispatching inference across ${N_GPUS} GPUs ..."

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
            IFS='|' read -r N SEQ_DIR PCT OUT_SUBDIR <<< "${WORK_QUEUE[$JOB_IDX]}"
            mkdir -p "$OUT_SUBDIR"

            PYTORCH_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES=$g \
            $PYTHON scripts/run_dust3r_inference.py \
                --sequence_dir    "$SEQ_DIR" \
                --dust3r_dir      "dust3r" \
                --n_frames        "$N" \
                --frame_pool_pct  "$PCT" \
                --output_dir      "$OUT_SUBDIR" \
                --chamfer_only \
                > "${OUT_SUBDIR}/inference.log" 2>&1 &

            GPU_PID[$g]=$!
            echo "  GPU $g → n=${N}  $(basename $SEQ_DIR)  pct=${PCT}  [job $((JOB_IDX+1))/${TOTAL_JOBS}]"
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
$PYTHON scripts/angle_range_exp/plot_angle_range_exp.py \
    --results_root "$OUTPUT_ROOT" \
    --n_min "$N_FRAMES_MIN" \
    --n_max "$N_FRAMES_MAX"

echo ""
echo "================================================================"
echo "Finished : $(date)"
echo "Plot     : ${OUTPUT_ROOT}/angle_range_compare.png"
echo "Transfer : rsync -av hminasyan@cluster.ysu.am:${OUTPUT_ROOT}/angle_range_compare.png outputs/dust3r/angle_range_exp/"
echo "================================================================"
