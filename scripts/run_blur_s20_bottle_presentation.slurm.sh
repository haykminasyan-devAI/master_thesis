#!/usr/bin/env bash
# One-off: blur sigma=20, n_frames=20, bottle/34_1397_4376
# Produces two predicted point clouds for side-by-side comparison:
#   - presentation/dust3r_pred.ply           (vanilla DUSt3R baseline)
#   - presentation/deblur_finetuned_pred.ply (DeblurDiNAT + DUSt3R, joint-finetuned σ∈{5,10,20})
# Plus per-pipeline metrics.txt and inference.log.

#SBATCH --job-name=blur_s20_f20_bottle_present
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/blur_s20_f20_bottle_present_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/blur_s20_f20_bottle_present_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
fi

CATEGORY="bottle"
SEQ_ID="34_1397_4376"
SIGMA=20
N_FRAMES=20

DUST3R_CKPT="${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
DEBLUR_REPO="${PROJECT_DIR}/DeblurDiNAT"
FT_CKPT="${PROJECT_DIR}/finetune_blur_runs/deblurdinat_dust3r_asds_224/joint_sigmas_5_10_20/checkpoint-best-val.pth"
BLUR_DIR="${PROJECT_DIR}/outputs/degraded_frames/${CATEGORY}/${SEQ_ID}/blur_s${SIGMA}"
GT_PLY="${PROJECT_DIR}/data/co3d_processed_10cat/${CATEGORY}/${SEQ_ID}/pointcloud.ply"

for f in "${DUST3R_CKPT}" "${FT_CKPT}" "${GT_PLY}"; do
  [[ -e "${f}" ]] || { echo "MISSING: ${f}" >&2; exit 1; }
done
[[ -d "${BLUR_DIR}" ]] || { echo "MISSING dir: ${BLUR_DIR}" >&2; exit 1; }

# Layout expected by run_inference.py: <seq>/images and <seq>/pointcloud.ply
SEQ_TMP="${PROJECT_DIR}/outputs/dust3r/_tmp_seq_blur_s${SIGMA}_${CATEGORY}_${SEQ_ID}"
rm -rf "${SEQ_TMP}"
mkdir -p "${SEQ_TMP}"
ln -sfn "${BLUR_DIR}" "${SEQ_TMP}/images"
ln -sfn "${GT_PLY}"   "${SEQ_TMP}/pointcloud.ply"

PRESENT_DIR="${PROJECT_DIR}/outputs/dust3r/blur_exp_224/blur_s${SIGMA}/${CATEGORY}_${SEQ_ID}/presentation"
DUST3R_DIR="${PRESENT_DIR}/_tmp_dust3r"
DEBLUR_DIR="${PRESENT_DIR}/_tmp_deblur_finetuned"
mkdir -p "${PRESENT_DIR}" "${DUST3R_DIR}" "${DEBLUR_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "  $(date) on $(hostname)"
echo "  sigma=${SIGMA}  n_frames=${N_FRAMES}  ${CATEGORY}/${SEQ_ID}"
echo "  Output: ${PRESENT_DIR}"
echo "============================================================"

# 1) Vanilla DUSt3R baseline
python3 finetune_blur/deblurdinat/run_inference.py \
  --sequence_dir "${SEQ_TMP}" \
  --dust3r_dir "${PROJECT_DIR}/dust3r" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --pipeline dust3r \
  --output_dir "${DUST3R_DIR}" \
  --image_size 224 \
  --n_frames "${N_FRAMES}" \
  --device cuda

# 2) Finetuned DeblurDiNAT + DUSt3R (joint sigmas 5/10/20 ckpt)
python3 finetune_blur/deblurdinat/run_inference.py \
  --sequence_dir "${SEQ_TMP}" \
  --dust3r_dir "${PROJECT_DIR}/dust3r" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --deblurdinat_repo "${DEBLUR_REPO}" \
  --pipeline deblur_finetuned \
  --finetuned_weights "${FT_CKPT}" \
  --output_dir "${DEBLUR_DIR}" \
  --image_size 224 \
  --n_frames "${N_FRAMES}" \
  --device cuda

# Move named files into presentation/ root
cp -f "${DUST3R_DIR}/pred_pointcloud.ply" "${PRESENT_DIR}/dust3r_pred.ply"
cp -f "${DEBLUR_DIR}/pred_pointcloud.ply" "${PRESENT_DIR}/deblur_finetuned_pred.ply"
cp -f "${DUST3R_DIR}/metrics.txt"        "${PRESENT_DIR}/dust3r_metrics.txt" 2>/dev/null || true
cp -f "${DEBLUR_DIR}/metrics.txt"        "${PRESENT_DIR}/deblur_finetuned_metrics.txt" 2>/dev/null || true
ln -sfn "${GT_PLY}" "${PRESENT_DIR}/gt_pointcloud.ply"

echo "============================================================"
echo "Predictions:"
ls -lh "${PRESENT_DIR}"
echo "Metrics (if present):"
[[ -f "${PRESENT_DIR}/dust3r_metrics.txt" ]] && cat "${PRESENT_DIR}/dust3r_metrics.txt" || true
echo "------------------------------------------------------------"
[[ -f "${PRESENT_DIR}/deblur_finetuned_metrics.txt" ]] && cat "${PRESENT_DIR}/deblur_finetuned_metrics.txt" || true
echo "============================================================"

# Cleanup temp staging
rm -rf "${SEQ_TMP}" "${DUST3R_DIR}" "${DEBLUR_DIR}"

echo "Done: $(date)"
