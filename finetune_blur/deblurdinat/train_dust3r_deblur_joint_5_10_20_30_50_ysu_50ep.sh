#!/usr/bin/env bash
# DeblurDiNAT + DUSt3R blur finetuning on YSU
# Joint sigmas: 5 10 20 30 50, 50 epochs, 224 setup.
#
# Usage:
#   cd /home/hminasyan/project_Hayk_Minasyan
#   sbatch --export=ALL finetune_blur/deblurdinat/train_dust3r_deblur_joint_5_10_20_30_50_ysu_50ep.sh
#
# Optional overrides:
#   sbatch --export=ALL,BLUR_SIGMAS="5 10 20 30 50",EPOCHS=50,NPROC=4 finetune_blur/deblurdinat/train_dust3r_deblur_joint_5_10_20_30_50_ysu_50ep.sh

#SBATCH --job-name=dust3r_deblur_5_10_20_30_50_50ep
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_deblur_5_10_20_30_50_50ep_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_deblur_5_10_20_30_50_50ep_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/finetune_blur/deblurdinat"
mkdir -p "${PROJECT_DIR}/logs"

# Portable env activation (YSU first, then local fallback)
if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
elif [[ -f "/home/hminasyan/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/home/hminasyan/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
elif [[ -f "/home/asds/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/home/asds/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
else
  echo "ERROR: Could not find co3d_env activation (weka venv or miniforge)."
  exit 1
fi
VENV_PYTHON="$(command -v python3)"

# Ensure CUDA NATTEN is available
"${VENV_PYTHON}" - <<'PY' || { echo "ERROR: NATTEN missing CUDA lib. Install wheel from https://whl.natten.org"; exit 1; }
from natten._libnatten import HAS_LIBNATTEN
import sys
sys.exit(0 if HAS_LIBNATTEN else 1)
PY

# YSU defaults (weka for large data/checkpoints)
: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d_processed_10cat}"
: "${BLUR_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames_10cat}"
: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
: "${DEBLURDINAT_WEIGHTS:=${DEBLURDINAT_REPO}/results/DeblurDiNATL/models/DeblurDiNATL.pth}"
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_blur_runs/deblurdinat_dust3r_ysu_224_joint_5_10_20_30_50_50ep}"

: "${BLUR_SIGMAS:=5 10 20 30 50}"
: "${FREEZE:=deblurdinat_only}"
: "${BATCH_SIZE:=1}"
: "${EPOCHS:=50}"
: "${LR:=2e-4}"
: "${WARMUP_EPOCHS:=0}"
: "${ETA_MIN:=1e-7}"
: "${GRAD_CLIP:=1.0}"
: "${WEIGHT_DECAY:=0.05}"
: "${RESOLUTION:=224}"
: "${AMP:=1}"
: "${GRAD_CHECKPOINT:=1}"
: "${NUM_WORKERS:=8}"
NPROC="${NPROC:-4}"

: "${WANDB_PROJECT:=master thesis}"
: "${WANDB_RUN_NAME:=deblurdinat_joint_5_10_20_30_50_ysu_50ep}"
: "${WANDB_ENTITY:=haykminasyan70-yerevan-state-university-ysu}"
export WANDB_ENTITY

# Optional: auto-submit 6-seq Chamfer evaluation after training completes successfully.
# This uses the new blur finetuned checkpoint and the existing YSU eval pipeline.
: "${AUTO_SUBMIT_EVAL_6SEQ:=1}"
: "${EVAL_SCRIPT:=${PROJECT_DIR}/evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer_ysu.sh}"
: "${EVAL_BLUR_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames}"
: "${EVAL_NOISE_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames}"
: "${EVAL_OUT_ROOT:=/mnt/weka/hminasyan/outputs/dust3r_eval_6seq_blur_noise_blurft_$(date +%Y%m%d_%H%M%S)}"

for f in "${DUST3R_CKPT}" "${DEBLURDINAT_WEIGHTS}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing file: $f"; exit 1; }
done
[[ -d "${DEBLURDINAT_REPO}" ]] || { echo "ERROR: DeblurDiNAT repo not found: ${DEBLURDINAT_REPO}"; exit 1; }

# 10-category sanity check: print all missing blur dirs before exiting.
# Uses finetune_blur/sequences_10cat.json by default.
: "${SEQ_JSON:=${PROJECT_DIR}/finetune_blur/sequences_10cat.json}"
CATEGORIES=(apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot)
declare -A SEQ_ID

if [[ ! -f "${SEQ_JSON}" ]]; then
  echo "ERROR: sequence manifest not found: ${SEQ_JSON}"
  echo "Run preprocessing first (or pass SEQ_JSON=/path/to/sequences_10cat.json)."
  exit 1
fi

for c in "${CATEGORIES[@]}"; do
  sid="$("${VENV_PYTHON}" -c "import json; d=json.load(open('${SEQ_JSON}')); print(d.get('${c}',''))")"
  if [[ -z "${sid}" ]]; then
    echo "ERROR: missing sequence for category '${c}' in ${SEQ_JSON}"
    exit 1
  fi
  SEQ_ID["${c}"]="${sid}"
done

MISSING=0
for c in "${CATEGORIES[@]}"; do
  sid="${SEQ_ID[$c]}"
  for s in ${BLUR_SIGMAS}; do
    d="${BLUR_ROOT}/${c}/${sid}/blur_s${s}"
    if [[ ! -d "${d}" ]]; then
      echo "MISSING: ${d}"
      MISSING=1
    fi
  done
done
if [[ "${MISSING}" -ne 0 ]]; then
  echo ""
  echo "ERROR: One or more blur folders are missing under BLUR_ROOT."
  echo "Generate them first, e.g.:"
  echo "  BLUR_SIGMAS=\"${BLUR_SIGMAS}\" bash finetune_blur/preprocess_for_training_10cat.sh"
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

echo "================================================================"
echo "YSU — DeblurDiNAT joint sigma(5,10,20,30,50) | 50 epochs | $(date) | $(hostname)"
echo "CO3D_ROOT        : ${CO3D_ROOT}"
echo "BLUR_ROOT        : ${BLUR_ROOT}"
echo "DUSt3R ckpt      : ${DUST3R_CKPT}"
echo "DeblurDiNAT init : ${DEBLURDINAT_WEIGHTS}"
echo "Output           : ${OUTPUT_DIR}"
echo "Sigmas           : ${BLUR_SIGMAS}"
echo "AUTO_SUBMIT_EVAL : ${AUTO_SUBMIT_EVAL_6SEQ}"
echo "================================================================"

JOINT_OUTPUT="${OUTPUT_DIR}/joint_sigmas_$(echo "${BLUR_SIGMAS}" | tr ' ' '_')"
mkdir -p "${JOINT_OUTPUT}"

"${VENV_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  "${SCRIPT_DIR}/train.py" \
  --co3d_root            "${CO3D_ROOT}" \
  --blur_root            "${BLUR_ROOT}" \
  --blur_sigmas          ${BLUR_SIGMAS} \
  --dust3r_ckpt          "${DUST3R_CKPT}" \
  --deblurdinat_repo     "${DEBLURDINAT_REPO}" \
  --deblurdinat_weights  "${DEBLURDINAT_WEIGHTS}" \
  --output_dir           "${JOINT_OUTPUT}" \
  --freeze               "${FREEZE}" \
  --batch_size           "${BATCH_SIZE}" \
  --epochs               "${EPOCHS}" \
  --num_workers          "${NUM_WORKERS}" \
  --lr                   "${LR}" \
  --weight_decay         "${WEIGHT_DECAY}" \
  --warmup_epochs        "${WARMUP_EPOCHS}" \
  --eta_min              "${ETA_MIN}" \
  --grad_clip            "${GRAD_CLIP}" \
  --resolution           "${RESOLUTION}" \
  --amp                  "${AMP}" \
  --grad_checkpoint      "${GRAD_CHECKPOINT}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"}

echo ""
BEST_CKPT="${JOINT_OUTPUT}/checkpoint-best-val.pth"
if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "WARNING: best checkpoint not found at ${BEST_CKPT}"
else
  echo "Best checkpoint: ${BEST_CKPT}"
fi

if [[ "${AUTO_SUBMIT_EVAL_6SEQ}" == "1" ]]; then
  if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "WARNING: eval script not found, skip auto submit: ${EVAL_SCRIPT}"
  elif [[ ! -f "${BEST_CKPT}" ]]; then
    echo "WARNING: best checkpoint missing, skip auto submit."
  else
    echo ""
    echo "Submitting 6-seq eval with new blur finetuned checkpoint..."
    echo "  EVAL_SCRIPT : ${EVAL_SCRIPT}"
    echo "  BLUR_ROOT   : ${EVAL_BLUR_ROOT}"
    echo "  NOISE_ROOT  : ${EVAL_NOISE_ROOT}"
    echo "  OUT_ROOT    : ${EVAL_OUT_ROOT}"
    sbatch --export=ALL,BLUR_FINETUNED_CKPT="${BEST_CKPT}",BLUR_ROOT="${EVAL_BLUR_ROOT}",NOISE_ROOT="${EVAL_NOISE_ROOT}",OUT_ROOT="${EVAL_OUT_ROOT}" "${EVAL_SCRIPT}"
  fi
fi

echo "All done: $(date)"
