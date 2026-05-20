#!/usr/bin/env bash
#SBATCH --job-name=eval_kd_enc_dust3r
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/eval_kd_encoder_dust3r_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/eval_kd_encoder_dust3r_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

CO3D_PROC="${CO3D_PROC:-/mnt/weka/hminasyan/data/co3d_processed}"
CO3D_RAW="${CO3D_RAW:-/mnt/weka/hminasyan/data/co3d}"
SPLIT="${SPLIT:-test}"
N_FRAMES="${N_FRAMES:-20}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ALIGN_NITER="${ALIGN_NITER:-300}"
DEVICE="${DEVICE:-cuda}"

DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
KD20_CKPT="${KD20_CKPT:-/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_base_dust3r_stage1_2gpu/student_lora_best.pth}"
KD50_CKPT="${KD50_CKPT:-/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_base_dust3r_stage1_2gpu_stride2_ep50/student_lora_best.pth}"

OUT_JSON="${OUT_JSON:-/mnt/weka/hminasyan/outputs/eval_kd_encoder_dust3r/chamfer_eval_kd_encoder_${SPLIT}_n${N_FRAMES}.json}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/weka/hminasyan/co3d_env/bin/python3}"

mkdir -p /mnt/weka/hminasyan/logs "$(dirname "${OUT_JSON}")"

echo "================================================================"
echo "Eval KD-Encoder DUSt3R Chamfer"
echo "================================================================"
echo "Started      : $(date)"
echo "Host         : $(hostname)"
echo "CO3D_PROC    : ${CO3D_PROC}"
echo "CO3D_RAW     : ${CO3D_RAW}"
echo "Split        : ${SPLIT}"
echo "n_frames     : ${N_FRAMES}"
echo "image_size   : ${IMAGE_SIZE}"
echo "align_niter  : ${ALIGN_NITER}"
echo "DUSt3R ckpt  : ${DUST3R_CKPT}"
echo "KD20 ckpt    : ${KD20_CKPT}"
echo "KD50 ckpt    : ${KD50_CKPT}"
echo "Output JSON  : ${OUT_JSON}"
echo "================================================================"

"${PYTHON_BIN}" "eval-KD-Encoder-DUSt3R/eval_kd_encoder_dust3r_chamfer.py" \
  --co3d_processed "${CO3D_PROC}" \
  --co3d_raw "${CO3D_RAW}" \
  --split "${SPLIT}" \
  --categories bottle cup donut teddybear couch toytrain \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --kd20_ckpt "${KD20_CKPT}" \
  --kd50_ckpt "${KD50_CKPT}" \
  --dark_gammas 1.5 2.2 \
  --n_frames "${N_FRAMES}" \
  --image_size "${IMAGE_SIZE}" \
  --align_niter "${ALIGN_NITER}" \
  --device "${DEVICE}" \
  --out_json "${OUT_JSON}"

echo "Done: $(date)"
echo "Result JSON: ${OUT_JSON}"

