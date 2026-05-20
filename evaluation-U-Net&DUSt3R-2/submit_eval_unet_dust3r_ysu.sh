#!/usr/bin/env bash
#SBATCH --job-name=eval_unet_d3r2
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/eval_unet_dust3r2_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/eval_unet_dust3r2_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

# Default: 8-category processed CO3D + `test` split (bottle/cup/…). Do NOT use test_10cat8 on
# co3d_processed_10cat8seq_fixed for these categories — that split lists the other 10 classes only.
CO3D_PROC="${CO3D_PROC:-/mnt/weka/hminasyan/data/co3d_processed}"
CO3D_RAW="${CO3D_RAW:-/mnt/weka/hminasyan/data/co3d}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
STUDENT_CKPT="${STUDENT_CKPT:-/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_restormer_frontend_1gpu/student_best.pth}"

SPLIT="${SPLIT:-test}"
N_FRAMES="${N_FRAMES:-20}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
DEVICE="${DEVICE:-cuda}"

OUT_JSON="${OUT_JSON:-/mnt/weka/hminasyan/outputs/eval_unet_dust3r2/chamfer_unet_dust3r2_${SPLIT}_n${N_FRAMES}_$(date +%Y%m%d_%H%M%S).json}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/weka/hminasyan/co3d_env/bin/python3}"

EVAL_SCRIPT="${PROJECT_DIR}/evaluation-U-Net&DUSt3R-2/eval_unet_dust3r_chamfer.py"

mkdir -p /mnt/weka/hminasyan/logs "$(dirname "${OUT_JSON}")"

echo "================================================================"
echo "Eval KD student + DUSt3R Chamfer (6 scenarios) — evaluation-U-Net&DUSt3R-2"
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

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
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
