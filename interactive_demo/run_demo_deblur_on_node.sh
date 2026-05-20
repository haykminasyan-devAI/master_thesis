#!/usr/bin/env bash
# Gradio demo for DeblurDiNAT + DUSt3R finetuned (Gaussian blur σ ∈ {5,10,20,30,(50)}).
# Use after: salloc/srun --gres=gpu:1
#
# Upload frames from e.g.:
#   outputs/viz_selections/teddybear_34_1479_4753_blur_s10/
#
# In the UI pick: "DeblurDiNAT + DUSt3R finetuned (σ ∈ {5,10,20,30})"

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
fi

CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/interactive_demo/demo_ckpts}"
: "${DUST3R_224_CKPT:=${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
: "${FT_5_10_20_30:=${CKPT_DIR}/joint_sigmas_5_10_20_30/checkpoint-best-val.pth}"
: "${FT_5_10_20_30_50:=${CKPT_DIR}/joint_sigmas_5_10_20_30_50/checkpoint-best-val.pth}"
: "${SERVER_PORT:=7860}"
: "${DEVICE:=cuda}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[demo] DeblurDiNAT+DUSt3R  host=$(hostname -s)"
echo "[demo] DUST3R_224=${DUST3R_224_CKPT}"
echo "[demo] joint_30=${FT_5_10_20_30}"
echo "[demo] port=${SERVER_PORT}"
echo ""
echo "Upload: ${PROJECT_DIR}/outputs/viz_selections/teddybear_34_1479_4753_blur_s10/"
echo ""

python "${PROJECT_DIR}/interactive_demo/demo_finetuned.py" \
  --dust3r_224_ckpt "${DUST3R_224_CKPT}" \
  --deblurdinat_repo "${DEBLURDINAT_REPO}" \
  --finetuned_5_10_20_30 "${FT_5_10_20_30}" \
  --finetuned_5_10_20_30_50 "${FT_5_10_20_30_50}" \
  --deblur_image_size 224 \
  --dust3r_ckpt "${DUST3R_224_CKPT}" \
  --image_size 224 \
  --device "${DEVICE}" \
  --server_port "${SERVER_PORT}" \
  --local_network
