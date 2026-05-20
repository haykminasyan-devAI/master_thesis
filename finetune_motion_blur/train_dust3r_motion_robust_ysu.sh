#!/usr/bin/env bash
set -euo pipefail

# DUSt3R motion-robust fine-tuning on YSU.
# Usage:
#   bash finetune_motion_blur/train_dust3r_motion_robust_ysu.sh

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
CO3D_ROOT="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/weka/hminasyan/finetune_motion_blur_runs/dust3r_motion_robust_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-4}" \
  finetune_motion_blur/train_dust3r_motion_robust.py \
  --co3d_root "${CO3D_ROOT}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --epochs "${EPOCHS:-20}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --resolution "${RESOLUTION:-224}" \
  --lr_main "${LR_MAIN:-1e-5}" \
  --lr_encoder_low "${LR_ENCODER_LOW:-1e-6}" \
  --encoder_blocks_train "${ENCODER_BLOCKS_TRAIN:-4}" \
  --unfreeze_full_encoder_epoch "${UNFREEZE_EPOCH:--1}" \
  --kernel_min "${KERNEL_MIN:-3}" \
  --kernel_max "${KERNEL_MAX:-9}" \
  --train_clean_only "${TRAIN_CLEAN_ONLY:-0}" \
  --amp "${AMP:-1}" \
  --print_freq "${PRINT_FREQ:-20}" \
  --seed "${SEED:-42}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"}
