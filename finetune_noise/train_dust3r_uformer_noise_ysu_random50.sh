#!/usr/bin/env bash
# Uformer + DUSt3R noise finetuning — YSU cluster
# Same hyperparameters as dust3r_noise_asds_471256 (lr=2e-4, wd=0.02, warmup=3, lambda_recon=0.05, ...)
# but: 50 epochs from scratch, random 80/10/10 or 80/20/0 split via --val_ratio/--test_ratio
# (471256 used ~80% train / 20% val category split → we use val_ratio=0.2, test_ratio=0)
#
# Usage:
#   export WANDB_API_KEY='...'   # required; no leading/trailing spaces
#   cd /home/hminasyan/project_Hayk_Minasyan
#   sbatch --export=ALL finetune_noise/train_dust3r_uformer_noise_ysu_random50.sh
# If the job lands on a bad GPU node and dies with CUDA Error 802 at torch.cuda.set_device,
# resubmit excluding that host, e.g.:
#   sbatch --exclude=gpu04 --export=ALL finetune_noise/train_dust3r_uformer_noise_ysu_random50.sh
# WANDB_ENTITY defaults to haykminasyan70-yerevan-state-university-ysu (override if needed).
#
#SBATCH --job-name=dust3r_noise_rand50
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_noise_rand50_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_noise_rand50_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Portable env (YSU weka venv first, then local miniforge fallbacks)
if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
elif [[ -f "/home/asds/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/home/asds/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
elif [[ -f "/home/hminasyan/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/home/hminasyan/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
else
  echo "ERROR: Could not find co3d_env activation."
  exit 1
fi

VENV_PYTHON="$(command -v python3)"

# Data / ckpts on YSU (override with sbatch --export if needed)
: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d_processed_10cat}"
: "${NOISE_ROOT:=/mnt/weka/hminasyan/outputs/noisy_frames_10cat}"
: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${UFORMER_REPO:=${PROJECT_DIR}/Uformer}"
: "${UFORMER_WEIGHTS:=${PROJECT_DIR}/Uformer/logs/denoising/SIDD/Uformer_B/models/model_best.pth}"

# Fresh run output (random split, 50 epochs) — lives on weka for space
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_noise_runs/uformer_dust3r_asds_224_recon_l05_lr2e4_wu3_wd02_random50}"

# Match job 471256
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
# ~80/20 train/val like the category-split experiment (no held-out test)
: "${VAL_RATIO:=0.2}"
: "${TEST_RATIO:=0.0}"
NPROC="${NPROC:-4}"

: "${WANDB_PROJECT:=master thesis}"
: "${WANDB_RUN_NAME:=uformer_noise_random50_lr2e4_wu3}"
: "${WANDB_ENTITY:=haykminasyan70-yerevan-state-university-ysu}"
export WANDB_ENTITY

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

echo "================================================================"
echo "YSU — Uformer (+ frozen DUSt3R) | random split | $(date) | $(hostname)"
echo "CO3D_ROOT      : ${CO3D_ROOT}"
echo "NOISE_ROOT     : ${NOISE_ROOT}"
echo "Noise sigmas   : 30 50 70"
echo "Split          : random (val_ratio=${VAL_RATIO}, test_ratio=${TEST_RATIO}, seed=${SEED})"
echo "DUSt3R ckpt    : ${DUST3R_CKPT}"
echo "Uformer ckpt   : ${UFORMER_WEIGHTS}"
echo "Output         : ${OUTPUT_DIR}"
echo "Hyperparams    : LR=${LR} WD=${WEIGHT_DECAY} warmup=${WARMUP_EPOCHS} lambda_recon=${LAMBDA_RECON} epochs=${EPOCHS}"
echo "W&B entity     : ${WANDB_ENTITY}"
echo "NPROC          : ${NPROC}"
echo "================================================================"

JOINT_OUTPUT="${OUTPUT_DIR}/joint_sigmas_30_50_70"
mkdir -p "${JOINT_OUTPUT}"

"${VENV_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  finetune_noise/train.py \
  --co3d_root "${CO3D_ROOT}" \
  --noise_root "${NOISE_ROOT}" \
  --noise_sigmas 30 50 70 \
  --random_split \
  --val_ratio "${VAL_RATIO}" \
  --test_ratio "${TEST_RATIO}" \
  --seed "${SEED}" \
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
