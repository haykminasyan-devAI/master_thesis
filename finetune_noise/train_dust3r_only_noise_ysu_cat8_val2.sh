#!/usr/bin/env bash
# DUSt3R-only noise finetuning on YSU (10 categories -> 8 train, 2 val_ood).
#
# Usage:
#   export WANDB_API_KEY='...'
#   cd /home/hminasyan/project_Hayk_Minasyan
#   sbatch --export=ALL finetune_noise/train_dust3r_only_noise_ysu_cat8_val2.sh

#SBATCH --job-name=dust3r_noise_cat8_val2
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_noise_cat8_val2_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_noise_cat8_val2_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

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

: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d_processed_10cat}"
: "${NOISE_ROOT:=/mnt/weka/hminasyan/outputs/noisy_frames_10cat}"
: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_noise_runs/dust3r_only_ysu_224_noise_cat8_val2}"

: "${TRAIN_CATEGORIES:=apple banana baseballbat baseballglove bicycle bowl broccoli cake}"
: "${VAL_CATEGORIES:=car carrot}"
: "${NOISE_SIGMAS:=30 50 70 80}"
: "${VAL_SOURCE_SPLIT:=test}"

: "${BATCH_SIZE:=2}"
: "${EPOCHS:=50}"
: "${LR:=1e-4}"
: "${WEIGHT_DECAY:=0.05}"
: "${WARMUP_EPOCHS:=5}"
: "${ETA_MIN:=1e-6}"
: "${GRAD_CLIP:=1.0}"
: "${RESOLUTION:=224}"
: "${AMP:=1}"
: "${NUM_WORKERS:=8}"
: "${VAL_EVERY:=1}"
: "${EARLY_STOP_PATIENCE:=0}"
: "${SEED:=0}"
NPROC="${NPROC:-4}"

: "${WANDB_PROJECT:=master thesis}"
: "${WANDB_RUN_NAME:=dust3r_only_noise_cat8_val2_ysu}"
: "${WANDB_ENTITY:=haykminasyan70-yerevan-state-university-ysu}"
export WANDB_ENTITY

[[ -f "${DUST3R_CKPT}" ]] || { echo "ERROR: missing checkpoint ${DUST3R_CKPT}"; exit 1; }
[[ -d "${CO3D_ROOT}" ]] || { echo "ERROR: missing CO3D_ROOT ${CO3D_ROOT}"; exit 1; }
[[ -d "${NOISE_ROOT}" ]] || { echo "ERROR: missing NOISE_ROOT ${NOISE_ROOT}"; exit 1; }

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"

echo "================================================================"
echo "YSU DUSt3R-only noise finetuning | $(date) | $(hostname)"
echo "CO3D_ROOT       : ${CO3D_ROOT}"
echo "NOISE_ROOT      : ${NOISE_ROOT}"
echo "DUSt3R ckpt     : ${DUST3R_CKPT}"
echo "Output          : ${OUTPUT_DIR}"
echo "Noise sigmas    : ${NOISE_SIGMAS}"
echo "Train categories: ${TRAIN_CATEGORIES}"
echo "Val categories  : ${VAL_CATEGORIES}"
echo "Val split source: ${VAL_SOURCE_SPLIT}"
echo "================================================================"

RUN_OUTPUT="${OUTPUT_DIR}/joint_sigmas_$(echo "${NOISE_SIGMAS}" | tr ' ' '_')"
mkdir -p "${RUN_OUTPUT}"

"${VENV_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  finetune_noise/train.py \
  --co3d_root "${CO3D_ROOT}" \
  --noise_root "${NOISE_ROOT}" \
  --noise_sigmas ${NOISE_SIGMAS} \
  --train_categories ${TRAIN_CATEGORIES} \
  --val_categories ${VAL_CATEGORIES} \
  --val_source_split "${VAL_SOURCE_SPLIT}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --dust3r_only \
  --output_dir "${RUN_OUTPUT}" \
  --freeze all \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --eta_min "${ETA_MIN}" \
  --grad_clip "${GRAD_CLIP}" \
  --resolution "${RESOLUTION}" \
  --amp "${AMP}" \
  --lambda_recon 0.0 \
  --val_every "${VAL_EVERY}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --seed "${SEED}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"}

echo "All done: $(date)"
