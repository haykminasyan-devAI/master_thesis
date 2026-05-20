#!/usr/bin/env bash
#SBATCH -J ifan_dust3r_defocus
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o ifan_dust3r_defocus_%j.log
#SBATCH -e ifan_dust3r_defocus_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
CO3D_ROOT="${CO3D_ROOT:-${PROJECT_DIR}/data/co3d_processed_10cat}"
DUST3R_CKPT="${DUST3R_CKPT:-${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
IFAN_REPO="${IFAN_REPO:-${PROJECT_DIR}/IFAN}"
IFAN_CKPT="${IFAN_CKPT:-${PROJECT_DIR}/IFAN/ckpt/IFAN.pytorch}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/finetune_defocus_runs/ifan_front_dust3r_freeze_14cat}"

# Provide exactly 14 categories here.
CATEGORIES="${CATEGORIES:-apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot hydrant toybus bottle cup}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
RESOLUTION="${RESOLUTION:-224}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
DEFOCUS_RADIUS="${DEFOCUS_RADIUS:-6}"
WANDB_PROJECT="${WANDB_PROJECT:-dust3r-ifan-defocus}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ifan_freeze_dust3r_14cat}"

source ~/.bashrc
conda activate co3d_env

mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"

torchrun --nproc_per_node=1 --master_port=29621 \
  finetune_defocus/train_ifan_front_dust3r_defocus.py \
  --co3d_root "${CO3D_ROOT}" \
  --categories ${CATEGORIES} \
  --defocus_radius "${DEFOCUS_RADIUS}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --ifan_repo "${IFAN_REPO}" \
  --ifan_ckpt "${IFAN_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --freeze ifan_only \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers 8 \
  --lr "${LR}" \
  --eta_min 1e-6 \
  --weight_decay 0.01 \
  --warmup_epochs 3 \
  --grad_clip 1.0 \
  --resolution "${RESOLUTION}" \
  --amp 1 \
  --val_ratio "${VAL_RATIO}" \
  --test_ratio "${TEST_RATIO}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}"
