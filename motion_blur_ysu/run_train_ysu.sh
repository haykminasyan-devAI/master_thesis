#!/usr/bin/env bash
set -euo pipefail
# Run on YSU after: source /mnt/weka/hminasyan/co3d_env/bin/activate
# Optional multi-GPU: torchrun --nproc_per_node=4 motion_blur_ysu/train_lora.py ...

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
# Pre-rename dir shadows CroCo if rsync did not use --delete
if [[ -f motion_blur_ysu/models/lora_cross_attn.py ]]; then
  echo "Removing stale motion_blur_ysu/models/ (use dust3r_lora/; conflicts with CroCo)."
  rm -rf motion_blur_ysu/models
fi
export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

CO3D_ROOT="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
OUT="${OUTPUT_DIR:-/mnt/weka/hminasyan/finetune_motion_blur_runs/lora_motion_${SLURM_JOB_ID:-local}}"

python motion_blur_ysu/train_lora.py \
  --co3d_root "${CO3D_ROOT}" \
  --dust3r_ckpt "${CKPT}" \
  --output_dir "${OUT}" \
  --split_train train_10cat8seq \
  --split_val val_10cat8seq \
  --epochs "${EPOCHS:-30}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --resolution "${RESOLUTION:-224}" \
  --lr "${LR:-1e-4}" \
  --eta_min "${ETA_MIN:-1e-5}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --amp "${AMP:-1}"

echo "Training finished. Checkpoints: ${OUT}/checkpoint_lora_best_val.pth"
