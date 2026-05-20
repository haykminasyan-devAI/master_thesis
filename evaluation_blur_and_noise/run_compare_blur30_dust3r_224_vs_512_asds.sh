#!/usr/bin/env bash
# Compare DUSt3R-only behavior at blur_s30 on two problematic sequences
# using 224 vs 512 checkpoints/image_size.
#
# Sequences: hydrant/106_12648_23157, toybus/111_13154_25988
# Pipeline : dust3r only (no DeblurDiNAT / Uformer)
#
# Submit:
#   cd /home/asds/project_Hayk_Minasyan
#   sbatch evaluation_blur_and_noise/run_compare_blur30_dust3r_224_vs_512_asds.sh

#SBATCH --job-name=cmp_b30_224_512
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/cmp_blur30_224_512_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/cmp_blur30_224_512_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ -f /home/asds/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/asds/miniforge3/etc/profile.d/conda.sh
  conda activate co3d_env
fi
PYTHON="${PYTHON:-python3}"

BLUR_ROOT="${BLUR_ROOT:-${PROJECT_DIR}/outputs/degraded_frames}"
CO3D_ROOT="${CO3D_ROOT:-${PROJECT_DIR}/data/co3d_processed_10cat}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/dust3r_blur30_cmp_224_512_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

CKPT_224="${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
CKPT_512="${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

echo "================================================================"
echo "DUSt3R blur_s30 comparison: 224 vs 512 | $(date) | $(hostname)"
echo "BLUR_ROOT : ${BLUR_ROOT}"
echo "CO3D_ROOT : ${CO3D_ROOT}"
echo "OUT_ROOT  : ${OUT_ROOT}"
echo "================================================================"

for f in "${CKPT_224}" "${CKPT_512}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing checkpoint: $f"; exit 1; }
done

declare -A SEQS
SEQS["hydrant"]="106_12648_23157"
SEQS["toybus"]="111_13154_25988"

run_one() {
  local variant="$1"   # 224 or 512
  local image_size="$2"
  local ckpt="$3"
  local cat="$4"
  local sid="$5"

  local blur_dir="${BLUR_ROOT}/${cat}/${sid}/blur_s30"
  local gt_ply="${CO3D_ROOT}/${cat}/${sid}/pointcloud.ply"
  local tmp="${OUT_ROOT}/_tmp_${variant}_${cat}_${sid}"
  local out="${OUT_ROOT}/${variant}/blur_s30_${cat}_${sid}_dust3r"

  if [[ ! -d "${blur_dir}" ]]; then
    echo "ERROR: missing blur_s30 dir: ${blur_dir}"
    exit 1
  fi
  if [[ ! -f "${gt_ply}" ]]; then
    echo "ERROR: missing GT pointcloud: ${gt_ply}"
    exit 1
  fi

  rm -rf "${tmp}"
  mkdir -p "${tmp}" "${out}"
  ln -sfn "${blur_dir}" "${tmp}/images"
  ln -sfn "${gt_ply}" "${tmp}/pointcloud.ply"

  echo ">>> ${variant} | ${cat}/${sid}"
  "${PYTHON}" "${PROJECT_DIR}/evaluation_blur_and_noise/deblurdinat_run_inference.py" \
    --sequence_dir "${tmp}" \
    --dust3r_dir "${PROJECT_DIR}/dust3r" \
    --dust3r_ckpt "${ckpt}" \
    --output_dir "${out}" \
    --image_size "${image_size}" \
    --n_frames 20 \
    --device cuda \
    --pipeline dust3r \
    --quiet

  rm -rf "${tmp}"
}

for cat in "${!SEQS[@]}"; do
  sid="${SEQS[$cat]}"
  run_one "224" "224" "${CKPT_224}" "${cat}" "${sid}"
  run_one "512" "512" "${CKPT_512}" "${cat}" "${sid}"
done

RESULT_CSV="${OUT_ROOT}/compare_blur30_dust3r_224_512.csv"
echo "variant,category,seq_id,chamfer_distance,cd_pred_to_gt,cd_gt_to_pred,n_pred_points,n_gt_points" > "${RESULT_CSV}"

for variant in 224 512; do
  for cat in "${!SEQS[@]}"; do
    sid="${SEQS[$cat]}"
    metrics="${OUT_ROOT}/${variant}/blur_s30_${cat}_${sid}_dust3r/metrics.txt"
    if [[ ! -f "${metrics}" ]]; then
      echo "${variant},${cat},${sid},NA,NA,NA,NA,NA" >> "${RESULT_CSV}"
      continue
    fi
    chamfer="$(rg "^chamfer_distance:" "${metrics}" | awk '{print $2}')"
    p2g="$(rg "^cd_pred_to_gt:" "${metrics}" | awk '{print $2}')"
    g2p="$(rg "^cd_gt_to_pred:" "${metrics}" | awk '{print $2}')"
    n_pred="$(rg "^n_pred_points:" "${metrics}" | awk '{print $2}')"
    n_gt="$(rg "^n_gt_points:" "${metrics}" | awk '{print $2}')"
    echo "${variant},${cat},${sid},${chamfer},${p2g},${g2p},${n_pred},${n_gt}" >> "${RESULT_CSV}"
  done
done

echo ""
echo "Done: $(date)"
echo "Compare CSV: ${RESULT_CSV}"
