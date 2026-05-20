#!/usr/bin/env bash
#SBATCH --job-name=eval_zdce_dust3r
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/eval_zerodce_dust3r_chamfer_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/eval_zerodce_dust3r_chamfer_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PROJECT_DIR}/KD-Zero-Reference:${PYTHONPATH:-}"

CO3D_PROC="${CO3D_PROC:-/mnt/weka/hminasyan/data/co3d_processed}"
CO3D_RAW="${CO3D_RAW:-/mnt/weka/hminasyan/data/co3d}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
# Default: epoch 249 = lowest val loss in log (same weights as student_best_val if you cp it)
STUDENT_CKPT="${STUDENT_CKPT:-/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_zerodce_uretinex/student_epoch_0249.pth}"
ZERODCE_ROOT="${ZERODCE_ROOT:-${PROJECT_DIR}/external/Zero-DCE}"

SPLIT="${SPLIT:-test}"
N_FRAMES="${N_FRAMES:-20}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ALIGN_NITER="${ALIGN_NITER:-300}"
DEVICE="${DEVICE:-cuda}"

OUT_JSON="${OUT_JSON:-/mnt/weka/hminasyan/outputs/eval_kd_zerodce_uretinex/chamfer_${SPLIT}_n${N_FRAMES}_$(date +%Y%m%d_%H%M%S).json}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/weka/hminasyan/co3d_env/bin/python3}"

mkdir -p /mnt/weka/hminasyan/logs "$(dirname "${OUT_JSON}")"

echo "================================================================"
echo "Eval Zero-DCE (KD) + DUSt3R Chamfer"
echo "================================================================"
echo "Started      : $(date)"
echo "Host         : $(hostname)"
echo "CO3D_PROC    : ${CO3D_PROC}"
echo "CO3D_RAW     : ${CO3D_RAW}"
echo "DUSt3R ckpt  : ${DUST3R_CKPT}"
echo "Student ckpt : ${STUDENT_CKPT}"
echo "Zero-DCE src : ${ZERODCE_ROOT}"
echo "Split        : ${SPLIT}"
echo "n_frames     : ${N_FRAMES}"
echo "image_size   : ${IMAGE_SIZE}"
echo "align_niter  : ${ALIGN_NITER}"
echo "Output JSON  : ${OUT_JSON}"
echo "================================================================"

"${PYTHON_BIN}" "${PROJECT_DIR}/EVAL-kd_zerodce_uretinex/eval_zerodce_dust3r_chamfer.py" \
  --co3d_processed "${CO3D_PROC}" \
  --co3d_raw "${CO3D_RAW}" \
  --split "${SPLIT}" \
  --categories bottle couch cup donut hydrant teddybear toybus toytrain \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --student_ckpt "${STUDENT_CKPT}" \
  --zerodce_root "${ZERODCE_ROOT}" \
  --n_frames "${N_FRAMES}" \
  --image_size "${IMAGE_SIZE}" \
  --align_niter "${ALIGN_NITER}" \
  --device "${DEVICE}" \
  --out_json "${OUT_JSON}" \
  "$@"

echo "Done: $(date)"
echo "Result JSON: ${OUT_JSON}"
