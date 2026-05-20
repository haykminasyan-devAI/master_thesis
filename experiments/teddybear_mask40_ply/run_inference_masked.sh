#!/usr/bin/env bash
# Prepare DUSt3R-aligned 40% patch-masked frames (15), then run DUSt3R → predicted.ply
# Usage: from repo root: bash experiments/teddybear_mask40_ply/run_inference_masked.sh
# Requires: working torch/torchvision + GPU (set --device cpu if needed).

set -euo pipefail

# Some shells (e.g. IDE) set CUDA_VISIBLE_DEVICES=-1, which hides all GPUs from PyTorch.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "-1" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_ROOT"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
MASKED_DIR="${EXP_DIR}/masked_images"
OUT_DIR="${EXP_DIR}/dust3r_output_masked"

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
  --output_dir "${OUT_DIR}" \
  --device cuda

echo "Point cloud: ${OUT_DIR}/predicted.ply"
