#!/usr/bin/env bash
# Run the Gradio demo on the current machine (expect a GPU if DEVICE=cuda).
# Use after: salloc/srun with --gres=gpu:1, or from submit_demo_slurm_asds.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]] && [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "co3d_env" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
fi

CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/interactive_demo/demo_ckpts}"
# Default: ViT-L 512 DPT + KD Restormer student (same base for vanilla and U-Net pipelines).
: "${DUST3R_CKPT:=${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
if [[ ! -f "${DUST3R_CKPT}" ]]; then
  DUST3R_CKPT="${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
fi
if [[ ! -f "${DUST3R_CKPT}" ]]; then
  DUST3R_CKPT="${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
fi
: "${KD_RESTORMER_STUDENT_CKPT:=${PROJECT_DIR}/restoration_kd_ysu/outputs_from_ysu/kd_restormer_frontend_1gpu/student_best.pth}"
if [[ ! -f "${KD_RESTORMER_STUDENT_CKPT}" ]]; then
  KD_RESTORMER_STUDENT_CKPT="${CKPT_DIR}/student_best.pth"
fi
: "${KD20_CKPT:=${CKPT_DIR}/student_lora_best_20ep.pth}"
: "${KD50_CKPT:=${CKPT_DIR}/student_lora_best_50ep.pth}"
: "${DUST3R_224_CKPT:=${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
: "${FT_5_10_20_30:=${CKPT_DIR}/joint_sigmas_5_10_20_30/checkpoint-best-val.pth}"
: "${FT_5_10_20_30_50:=${CKPT_DIR}/joint_sigmas_5_10_20_30_50/checkpoint-best-val.pth}"
: "${IMAGE_SIZE:=512}"
: "${SERVER_PORT:=7860}"
: "${DEVICE:=cuda}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[demo] host=$(hostname -s)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[demo] port=${SERVER_PORT}  device=${DEVICE}"
echo ""

python "${PROJECT_DIR}/interactive_demo/demo_finetuned.py" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --dust3r_224_ckpt "${DUST3R_224_CKPT}" \
  --deblurdinat_repo "${DEBLURDINAT_REPO}" \
  --finetuned_5_10_20_30 "${FT_5_10_20_30}" \
  --finetuned_5_10_20_30_50 "${FT_5_10_20_30_50}" \
  --kd_restormer_student_ckpt "${KD_RESTORMER_STUDENT_CKPT}" \
  --kd20_ckpt "${KD20_CKPT}" \
  --kd50_ckpt "${KD50_CKPT}" \
  --image_size "${IMAGE_SIZE}" \
  --deblur_image_size 224 \
  --device "${DEVICE}" \
  --server_port "${SERVER_PORT}" \
  --local_network
