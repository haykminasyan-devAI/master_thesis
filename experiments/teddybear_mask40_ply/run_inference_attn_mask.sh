#!/usr/bin/env bash
# Same as run_inference_masked.sh but passes patch attention key masks so blacked-out
# patches are not used as attention keys (encoder self-attn + decoder cross-attn).
# Requires: prepare_dust3r_aligned_masked.py (writes masked_images + masked_images_patch_masks).

set -euo pipefail

# Some shells (e.g. IDE) set CUDA_VISIBLE_DEVICES=-1, which hides all GPUs from PyTorch.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "-1" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_ROOT"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
MASKED_DIR="${EXP_DIR}/masked_images"
MASK_NPY_DIR="${MASKED_DIR}_patch_masks"
OUT_DIR="${EXP_DIR}/dust3r_output_attn_key_masked"

SEQ="data/co3d/teddybear/101_11758_21048"
N_FRAMES=15
MASK_RATIO=0.4

python "${EXP_DIR}/prepare_dust3r_aligned_masked.py" \
  --images_dir "${SEQ}/images" \
  --out_dir "${MASKED_DIR}" \
  --n_frames "${N_FRAMES}" \
  --mask_ratio "${MASK_RATIO}" \
  --seed 42

python scripts/run_dust3r_inference.py \
  --sequence_dir "${SEQ}" \
  --dust3r_dir dust3r \
  --n_frames "${N_FRAMES}" \
  --n_masked "${N_FRAMES}" \
  --masked_dir "${MASKED_DIR}" \
  --mask_ratio "${MASK_RATIO}" \
  --attn_mask_npy_dir "${MASK_NPY_DIR}" \
  --output_dir "${OUT_DIR}" \
  --device cuda

echo "PLY (attention-masked keys): ${OUT_DIR}/predicted.ply"
