#!/bin/bash
# One-GPU smoke test: teddybear, n_frames=6, 25% patch keys masked.

set -euo pipefail
cd /home/asds/project_Hayk_Minasyan

SEQ="data/co3d/teddybear/101_11758_21048"
MASK_DIR="outputs/patch_attn_masks/teddybear/101_11758_21048/mask_25pct_local"
OUT="outputs/dust3r/patch_attn_mask_exp_local/teddybear_frames06"

mkdir -p "$MASK_DIR"
python3 scripts/patch_attn_mask_exp/generate_patch_attn_masks.py \
    --images_dir "${SEQ}/images" \
    --output_dir "$MASK_DIR" \
    --mask_ratio 0.25 \
    --seed 0

python3 scripts/run_dust3r_inference.py \
    --sequence_dir "$SEQ" \
    --dust3r_dir   "dust3r" \
    --n_frames     6 \
    --n_masked     0 \
    --output_dir   "$OUT" \
    --mask_ratio   0.25 \
    --attn_mask_npy_dir "$MASK_DIR"

echo "Done → $OUT/metrics.txt"
