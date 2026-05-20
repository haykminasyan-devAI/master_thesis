#!/usr/bin/env bash
set -euo pipefail
# Final test Chamfer (requires processed test categories under CO3D_PROCESSED + pointclouds).

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
if [[ -f motion_blur_ysu/models/lora_cross_attn.py ]]; then
  echo "Removing stale motion_blur_ysu/models/ (use dust3r_lora/; conflicts with CroCo)."
  rm -rf motion_blur_ysu/models
fi
export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

CO3D_PROC="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
CO3D_RAW="${CO3D_RAW:-/mnt/weka/hminasyan/data/co3d}"
CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
LORA_CKPT="${LORA_CKPT:?Set LORA_CKPT to checkpoint_lora_best_val.pth}"

OUT="${EVAL_JSON:-/mnt/weka/hminasyan/outputs/chamfer_test_lora.json}"

python motion_blur_ysu/eval_report_chamfer.py \
  --co3d_processed "${CO3D_PROC}" \
  --co3d_raw "${CO3D_RAW}" \
  --split test_10cat8seq \
  --dust3r_ckpt "${CKPT}" \
  --lora_ckpt "${LORA_CKPT}" \
  --n_frames 20 \
  --image_size 224 \
  --out_json "${OUT}"

echo "Wrote ${OUT}"
