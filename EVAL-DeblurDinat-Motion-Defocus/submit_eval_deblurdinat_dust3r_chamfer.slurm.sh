#!/usr/bin/env bash
#SBATCH --job-name=eval_dinat_d3r
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/eval_deblurdinat_dust3r_chamfer_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/eval_deblurdinat_dust3r_chamfer_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

DEBLUR_DIR="${PROJECT_DIR}/finetuning Motion&Defocus/deblurdinat"
export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${DEBLUR_DIR}:${PYTHONPATH:-}"

CO3D_PROC="${CO3D_PROC:-/mnt/weka/hminasyan/data/co3d_processed}"
CO3D_RAW="${CO3D_RAW:-/mnt/weka/hminasyan/data/co3d}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
DEBLURDINAT_REPO="${DEBLURDINAT_REPO:-${PROJECT_DIR}/DeblurDiNAT}"
DEBLURDINAT_WEIGHTS="${DEBLURDINAT_WEIGHTS:-${DEBLURDINAT_REPO}/results/DeblurDiNATL/models/DeblurDiNATL.pth}"
FINETUNED_CKPT="${FINETUNED_CKPT:-/mnt/weka/hminasyan/finetune_motion_blur_runs/deblurdinat_motion_defocus_512_4cat/checkpoint-best-val.pth}"

SPLIT="${SPLIT:-test}"
N_FRAMES="${N_FRAMES:-20}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ALIGN_NITER="${ALIGN_NITER:-300}"
MOTION_PROB="${MOTION_PROB:-0.5}"
DEVICE="${DEVICE:-cuda}"

OUT_JSON="${OUT_JSON:-/mnt/weka/hminasyan/outputs/eval_deblurdinat_motion_defocus/chamfer_${SPLIT}_n${N_FRAMES}_$(date +%Y%m%d_%H%M%S).json}"
# DUSt3R global_aligner needs `roma` (dust3r/cloud_opt); co3d_env usually has it; deblurdinat_env may not.
PYTHON_BIN="${PYTHON_BIN:-/mnt/weka/hminasyan/co3d_env/bin/python3}"

mkdir -p /mnt/weka/hminasyan/logs "$(dirname "${OUT_JSON}")"

echo "================================================================"
echo "Eval DeblurDiNAT + DUSt3R Chamfer (6 scenarios)"
echo "================================================================"
echo "Started       : $(date)"
echo "Host          : $(hostname)"
echo "CO3D_PROC     : ${CO3D_PROC}"
echo "CO3D_RAW      : ${CO3D_RAW}"
echo "DUSt3R ckpt   : ${DUST3R_CKPT}"
echo "DeblurDiNAT   : ${DEBLURDINAT_WEIGHTS}"
echo "Finetuned ckpt: ${FINETUNED_CKPT}"
echo "Split         : ${SPLIT}"
echo "motion_prob   : ${MOTION_PROB}"
echo "n_frames      : ${N_FRAMES}"
echo "Output JSON   : ${OUT_JSON}"
echo "================================================================"

"${PYTHON_BIN}" "${PROJECT_DIR}/EVAL-DeblurDinat-Motion-Defocus/eval_deblurdinat_dust3r_chamfer.py" \
  --co3d_processed "${CO3D_PROC}" \
  --co3d_raw "${CO3D_RAW}" \
  --split "${SPLIT}" \
  --categories bottle couch cup donut hydrant teddybear toybus toytrain \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --deblurdinat_repo "${DEBLURDINAT_REPO}" \
  --deblurdinat_weights "${DEBLURDINAT_WEIGHTS}" \
  --finetuned_ckpt "${FINETUNED_CKPT}" \
  --motion_prob "${MOTION_PROB}" \
  --n_frames "${N_FRAMES}" \
  --image_size "${IMAGE_SIZE}" \
  --align_niter "${ALIGN_NITER}" \
  --device "${DEVICE}" \
  --out_json "${OUT_JSON}" \
  "$@"

echo "Done: $(date)"
echo "Result JSON: ${OUT_JSON}"
