#!/usr/bin/env bash
#SBATCH --job-name=dust3r_noise_asds
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/dust3r_noise_asds_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/dust3r_noise_asds_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env

VENV_PYTHON="$(command -v python3)"

: "${CO3D_ROOT:=${PROJECT_DIR}/data/co3d_processed_10cat}"
: "${NOISE_ROOT:=${PROJECT_DIR}/outputs/noisy_frames_10cat}"
: "${NOISE_SIGMAS:=30 50 70}"
: "${DUST3R_CKPT:=${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${UFORMER_REPO:=${PROJECT_DIR}/Uformer}"
: "${UFORMER_WEIGHTS:=${PROJECT_DIR}/Uformer/logs/denoising/SIDD/Uformer_B/models/model_best.pth}"
: "${OUTPUT_DIR:=${PROJECT_DIR}/finetune_noise_runs/uformer_dust3r_asds_224_recon_l05_lr5e5_wu3_wd5e4}"

# 10 categories -> 8 train, 2 val
: "${TRAIN_CATEGORIES:=apple banana baseballbat baseballglove bicycle bowl broccoli cake}"
: "${VAL_CATEGORIES:=car carrot}"

: "${FREEZE:=uformer_only}"
: "${BATCH_SIZE:=2}"
: "${EPOCHS:=30}"
: "${LR:=2e-4}"
: "${WEIGHT_DECAY:=0.02}"
: "${WARMUP_EPOCHS:=0}"
: "${ETA_MIN:=1e-6}"
: "${GRAD_CLIP:=1.0}"
: "${RESOLUTION:=224}"
: "${AMP:=1}"
: "${NUM_WORKERS:=8}"
: "${VAL_EVERY:=2}"
: "${EARLY_STOP_PATIENCE:=0}"
: "${LAMBDA_RECON:=0.05}"
NPROC="${NPROC:-4}"

: "${WANDB_PROJECT:=master thesis}"
: "${WANDB_RUN_NAME:=}"

if [[ ! -f "${DUST3R_CKPT}" ]]; then
  echo "ERROR: DUSt3R checkpoint not found: ${DUST3R_CKPT}"
  exit 1
fi
if [[ ! -f "${UFORMER_WEIGHTS}" ]]; then
  echo "ERROR: Uformer weights not found: ${UFORMER_WEIGHTS}"
  exit 1
fi
if [[ ! -d "${UFORMER_REPO}" ]]; then
  echo "ERROR: Uformer repo not found: ${UFORMER_REPO}"
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${UFORMER_REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

echo "================================================================"
echo "ASDS — Uformer (+ frozen DUSt3R) | $(date) | $(hostname)"
echo "CO3D_ROOT      : ${CO3D_ROOT}"
echo "NOISE_ROOT     : ${NOISE_ROOT}"
echo "Noise sigmas   : ${NOISE_SIGMAS}"
echo "Train categories: ${TRAIN_CATEGORIES}"
echo "Val categories  : ${VAL_CATEGORIES}"
echo "DUSt3R ckpt    : ${DUST3R_CKPT}"
echo "Uformer ckpt   : ${UFORMER_WEIGHTS}"
echo "Output         : ${OUTPUT_DIR}"
echo "NPROC          : ${NPROC}"
echo "================================================================"

JOINT_OUTPUT="${OUTPUT_DIR}/joint_sigmas_$(echo "${NOISE_SIGMAS}" | tr ' ' '_')"
mkdir -p "${JOINT_OUTPUT}"

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
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --uformer_repo "${UFORMER_REPO}" \
  --uformer_weights "${UFORMER_WEIGHTS}" \
  --output_dir "${JOINT_OUTPUT}" \
  --freeze "${FREEZE}" \
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
  --lambda_recon "${LAMBDA_RECON}" \
  --val_every "${VAL_EVERY}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"}

echo ""
echo "All done: $(date)"
