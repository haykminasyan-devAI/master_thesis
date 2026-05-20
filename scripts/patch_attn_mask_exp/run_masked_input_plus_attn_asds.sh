#!/bin/bash
#SBATCH --job-name=mask_input_attn
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mask_input_attn_%j.log
#SBATCH --error=logs/mask_input_attn_%j.err

set -euo pipefail

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
cd /home/asds/project_Hayk_Minasyan
mkdir -p logs

# Use current 6-seq list (bottle,cup,donut,teddybear,couch,toytrain)
source evaluation_blur_and_noise/sequences_6eval.inc.sh

MASK_RATIOS=(0.10 0.25 0.50)
MASK_TAGS=(mask_10pct mask_25pct mask_50pct)

N_FRAMES_MIN=2
N_FRAMES_MAX=20
N_GPUS=4

SEQ_BASE="data/co3d"
MASKED_ROOT="outputs/dust3r/attn_mask/masked_inputs"
ATTN_NPY_ROOT="outputs/dust3r/attn_mask/patch_masks"
OUTPUT_ROOT="outputs/dust3r/attn_mask/masked_input_plus_attn_exp"

echo "================================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "SEQ_BASE    : ${SEQ_BASE}"
echo "OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo "================================================================"

echo ""
echo ">>> Generate masked images + patch-attn masks (all frames) ..."
for CAT in "${!SEQUENCES[@]}"; do
  SID="${SEQUENCES[$CAT]}"
  IMG_DIR="${SEQ_BASE}/${CAT}/${SID}/images"
  if [[ ! -d "${IMG_DIR}" ]]; then
    echo "  [skip] missing images dir: ${IMG_DIR}"
    continue
  fi

  N_ALL=$(ls -1 "${IMG_DIR}"/*.jpg 2>/dev/null | wc -l)
  if [[ "${N_ALL}" -eq 0 ]]; then
    echo "  [skip] no jpg frames in: ${IMG_DIR}"
    continue
  fi

  for i in "${!MASK_RATIOS[@]}"; do
    TAG="${MASK_TAGS[$i]}"
    R="${MASK_RATIOS[$i]}"
    OUT_IMG="${MASKED_ROOT}/${CAT}/${SID}/${TAG}"
    OUT_NPY="${ATTN_NPY_ROOT}/${CAT}/${SID}/${TAG}"
    mkdir -p "${OUT_IMG}" "${OUT_NPY}"
    python3 experiments/teddybear_mask40_ply/prepare_dust3r_aligned_masked.py \
      --images_dir "${IMG_DIR}" \
      --out_dir "${OUT_IMG}" \
      --mask_npy_dir "${OUT_NPY}" \
      --n_frames "${N_ALL}" \
      --mask_ratio "${R}" \
      --seed 42 \
      --long_edge 512 \
      --patch_size 16
  done
done

echo ""
echo ">>> Build inference queue ..."
WORK_QUEUE=()
for N in $(seq ${N_FRAMES_MIN} ${N_FRAMES_MAX}); do
  for CAT in "${!SEQUENCES[@]}"; do
    SID="${SEQUENCES[$CAT]}"
    SEQ_DIR="${SEQ_BASE}/${CAT}/${SID}"
    for i in "${!MASK_RATIOS[@]}"; do
      TAG="${MASK_TAGS[$i]}"
      M_DIR="${MASKED_ROOT}/${CAT}/${SID}/${TAG}"
      A_DIR="${ATTN_NPY_ROOT}/${CAT}/${SID}/${TAG}"
      OUT="${OUTPUT_ROOT}/${TAG}/${CAT}_${SID}/frames_$(printf '%02d' "${N}")"
      if [[ ! -f "${OUT}/metrics.txt" ]]; then
        WORK_QUEUE+=("${N}|${SEQ_DIR}|${M_DIR}|${A_DIR}|${OUT}|${MASK_RATIOS[$i]}")
      fi
    done
  done
done

TOTAL=${#WORK_QUEUE[@]}
echo "  Total runs: ${TOTAL}"

echo ""
echo ">>> DUSt3R inference (${N_GPUS} GPUs) ..."
declare -a GPU_PID
for ((g=0; g<N_GPUS; g++)); do GPU_PID[$g]=0; done
COMPLETED=0
JOB_IDX=0

while [[ ${JOB_IDX} -lt ${TOTAL} || ${COMPLETED} -lt ${TOTAL} ]]; do
  for ((g=0; g<N_GPUS; g++)); do
    PID=${GPU_PID[$g]}
    if [[ ${PID} -ne 0 ]] && ! kill -0 ${PID} 2>/dev/null; then
      wait ${PID} || true
      COMPLETED=$((COMPLETED + 1))
      echo "    [${COMPLETED}/${TOTAL}] GPU ${g} done"
      GPU_PID[$g]=0
      PID=0
    fi
    if [[ ${PID} -eq 0 && ${JOB_IDX} -lt ${TOTAL} ]]; then
      IFS='|' read -r N SEQ_DIR M_DIR A_DIR OUT_SUBDIR RATIO <<< "${WORK_QUEUE[$JOB_IDX]}"
      mkdir -p "${OUT_SUBDIR}"
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES=${g} \
      python3 scripts/run_dust3r_inference.py \
        --sequence_dir "${SEQ_DIR}" \
        --dust3r_dir "dust3r" \
        --n_frames "${N}" \
        --n_masked "${N}" \
        --masked_dir "${M_DIR}" \
        --attn_mask_npy_dir "${A_DIR}" \
        --mask_ratio "${RATIO}" \
        --output_dir "${OUT_SUBDIR}" \
        > "${OUT_SUBDIR}/inference.log" 2>&1 &
      GPU_PID[$g]=$!
      echo "  GPU ${g} → n=${N} $(basename "${SEQ_DIR}") $(basename "${M_DIR}") [job $((JOB_IDX+1))/${TOTAL}]"
      JOB_IDX=$((JOB_IDX + 1))
    fi
  done
  sleep 5
done

echo ""
echo ">>> Plot ..."
python3 scripts/patch_attn_mask_exp/plot_patch_attn_mask_exp.py \
  --results_root "${OUTPUT_ROOT}" \
  --n_min "${N_FRAMES_MIN}" \
  --n_max "${N_FRAMES_MAX}"

echo ""
echo "================================================================"
echo "Finished : $(date)"
echo "Plot     : ${OUTPUT_ROOT}/patch_attn_mask_compare.png"
echo "================================================================"
