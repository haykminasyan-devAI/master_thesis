#!/usr/bin/env bash
# Uformer + DUSt3R noise finetuning on YSU
# Joint sigmas: 30 50 70 80, 50 epochs, 224 setup.
#
# Usage:
#   cd /home/hminasyan/project_Hayk_Minasyan
#   sbatch --export=ALL finetune_noise/train_dust3r_uformer_noise_joint_30_50_70_80_ysu_50ep.sh

#SBATCH --job-name=dust3r_noise_30_50_70_80_50ep
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_noise_30_50_70_80_50ep_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_noise_30_50_70_80_50ep_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/finetune_noise"
mkdir -p "${PROJECT_DIR}/logs"

# Portable env activation (YSU first)
if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
elif [[ -f "/home/hminasyan/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/home/hminasyan/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
else
  echo "ERROR: Could not find co3d_env activation."
  exit 1
fi
VENV_PYTHON="$(command -v python3)"

# YSU defaults (weka for large data/checkpoints)
: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d_processed_10cat}"
: "${NOISE_ROOT:=/mnt/weka/hminasyan/outputs/noisy_frames_10cat}"
: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${UFORMER_REPO:=${PROJECT_DIR}/Uformer}"
: "${UFORMER_WEIGHTS:=${UFORMER_REPO}/logs/denoising/SIDD/Uformer_B/models/model_best.pth}"
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_noise_runs/uformer_dust3r_ysu_224_joint_30_50_70_80_50ep}"

: "${NOISE_SIGMAS:=30 50 70 80}"
: "${FREEZE:=uformer_only}"
: "${BATCH_SIZE:=2}"
: "${EPOCHS:=50}"
: "${LR:=2e-4}"
: "${WEIGHT_DECAY:=0.02}"
: "${WARMUP_EPOCHS:=3}"
: "${ETA_MIN:=1e-6}"
: "${GRAD_CLIP:=1.0}"
: "${RESOLUTION:=224}"
: "${AMP:=1}"
: "${NUM_WORKERS:=8}"
: "${VAL_EVERY:=2}"
: "${EARLY_STOP_PATIENCE:=0}"
: "${LAMBDA_RECON:=0.05}"
: "${SEED:=0}"
: "${VAL_RATIO:=0.2}"
: "${TEST_RATIO:=0.0}"
NPROC="${NPROC:-4}"

: "${WANDB_PROJECT:=master thesis}"
: "${WANDB_RUN_NAME:=uformer_noise_joint_30_50_70_80_ysu_50ep}"
: "${WANDB_ENTITY:=haykminasyan70-yerevan-state-university-ysu}"
export WANDB_ENTITY

# Optional: auto-submit 6-seq Chamfer evaluation after successful training.
: "${AUTO_SUBMIT_EVAL_6SEQ:=1}"
: "${EVAL_SCRIPT:=${PROJECT_DIR}/evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer_ysu.sh}"
: "${EVAL_BLUR_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames}"
: "${EVAL_NOISE_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames}"
: "${EVAL_OUT_ROOT:=/mnt/weka/hminasyan/outputs/dust3r_eval_6seq_blur_noise_noisefinetune_$(date +%Y%m%d_%H%M%S)}"

for f in "${DUST3R_CKPT}" "${UFORMER_WEIGHTS}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing file: $f"; exit 1; }
done
[[ -d "${UFORMER_REPO}" ]] || { echo "ERROR: Uformer repo not found: ${UFORMER_REPO}"; exit 1; }

# 10-category sanity check: verify noisy folders exist for all requested sigmas
: "${SEQ_JSON:=${PROJECT_DIR}/finetune_blur/sequences_10cat.json}"
CATEGORIES=(apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot)
declare -A SEQ_ID

if [[ ! -f "${SEQ_JSON}" ]]; then
  echo "ERROR: sequence manifest not found: ${SEQ_JSON}"
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
  for s in ${NOISE_SIGMAS}; do
    d="${NOISE_ROOT}/${c}/${sid}/noise_s${s}"
    if [[ ! -d "${d}" ]]; then
      echo "MISSING: ${d}"
      MISSING=1
    fi
  done
done
if [[ "${MISSING}" -ne 0 ]]; then
  echo ""
  echo "ERROR: Missing noisy folders under NOISE_ROOT for requested sigmas."
  echo "Generate them first (e.g. preprocess_noise_10cat_asds.sh adapted for YSU)."
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${UFORMER_REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"

echo "================================================================"
echo "YSU — Uformer+DUSt3R joint noise(30,50,70,80) | 50 epochs | $(date) | $(hostname)"
echo "CO3D_ROOT      : ${CO3D_ROOT}"
echo "NOISE_ROOT     : ${NOISE_ROOT}"
echo "DUSt3R ckpt    : ${DUST3R_CKPT}"
echo "Uformer init   : ${UFORMER_WEIGHTS}"
echo "Output         : ${OUTPUT_DIR}"
echo "Noise sigmas   : ${NOISE_SIGMAS}"
echo "AUTO_SUBMIT_EVAL : ${AUTO_SUBMIT_EVAL_6SEQ}"
echo "================================================================"

JOINT_OUTPUT="${OUTPUT_DIR}/joint_sigmas_$(echo "${NOISE_SIGMAS}" | tr ' ' '_')"
mkdir -p "${JOINT_OUTPUT}"

"${VENV_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  "${SCRIPT_DIR}/train.py" \
  --co3d_root            "${CO3D_ROOT}" \
  --noise_root           "${NOISE_ROOT}" \
  --noise_sigmas         ${NOISE_SIGMAS} \
  --dust3r_ckpt          "${DUST3R_CKPT}" \
  --uformer_repo         "${UFORMER_REPO}" \
  --uformer_weights      "${UFORMER_WEIGHTS}" \
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
  --val_every            "${VAL_EVERY}" \
  --early_stop_patience  "${EARLY_STOP_PATIENCE}" \
  --lambda_recon         "${LAMBDA_RECON}" \
  --seed                 "${SEED}" \
  --val_ratio            "${VAL_RATIO}" \
  --test_ratio           "${TEST_RATIO}" \
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
    echo "Submitting 6-seq eval with new noise finetuned checkpoint..."
    echo "  EVAL_SCRIPT : ${EVAL_SCRIPT}"
    echo "  BLUR_ROOT   : ${EVAL_BLUR_ROOT}"
    echo "  NOISE_ROOT  : ${EVAL_NOISE_ROOT}"
    echo "  OUT_ROOT    : ${EVAL_OUT_ROOT}"
    sbatch --export=ALL,NOISE_FINETUNED_CKPT="${BEST_CKPT}",BLUR_ROOT="${EVAL_BLUR_ROOT}",NOISE_ROOT="${EVAL_NOISE_ROOT}",OUT_ROOT="${EVAL_OUT_ROOT}" "${EVAL_SCRIPT}"
  fi
fi

echo "All done: $(date)"
