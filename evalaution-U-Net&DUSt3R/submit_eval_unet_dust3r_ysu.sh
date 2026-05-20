#!/usr/bin/env bash
#SBATCH --job-name=eval_unet_dust3r
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/eval_unet_dust3r_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/eval_unet_dust3r_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

CO3D_PROC="${CO3D_PROC:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
CO3D_RAW="${CO3D_RAW:-/mnt/weka/hminasyan/data/co3d}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
STUDENT_CKPT="${STUDENT_CKPT:-/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_restormer_frontend_1gpu/student_best.pth}"

SPLIT="${SPLIT:-test_10cat8}"
N_FRAMES="${N_FRAMES:-20}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
DEVICE="${DEVICE:-cuda}"

OUT_JSON="${OUT_JSON:-/mnt/weka/hminasyan/outputs/eval_unet_dust3r/chamfer_unet_dust3r_${SPLIT}_n${N_FRAMES}.json}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/weka/hminasyan/co3d_env/bin/python3}"

mkdir -p /mnt/weka/hminasyan/logs "$(dirname "${OUT_JSON}")"

echo "================================================================"
echo "Eval U-Net + DUSt3R Chamfer"
echo "================================================================"
echo "Started      : $(date)"
echo "Host         : $(hostname)"
echo "CO3D_PROC    : ${CO3D_PROC}"
echo "CO3D_RAW     : ${CO3D_RAW}"
echo "DUSt3R ckpt  : ${DUST3R_CKPT}"
echo "Student ckpt : ${STUDENT_CKPT}"
echo "Split        : ${SPLIT}"
echo "n_frames     : ${N_FRAMES}"
echo "image_size   : ${IMAGE_SIZE}"
echo "Output JSON  : ${OUT_JSON}"
echo "================================================================"

"${PYTHON_BIN}" "evalaution-U-Net&DUSt3R/eval_unet_dust3r_chamfer.py" \
  --co3d_processed "${CO3D_PROC}" \
  --co3d_raw "${CO3D_RAW}" \
  --split "${SPLIT}" \
  --categories bottle cup donut teddybear couch toytrain \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --student_ckpt "${STUDENT_CKPT}" \
  --n_frames "${N_FRAMES}" \
  --image_size "${IMAGE_SIZE}" \
  --device "${DEVICE}" \
  --out_json "${OUT_JSON}"

echo "Done: $(date)"
echo "Result JSON: ${OUT_JSON}"
