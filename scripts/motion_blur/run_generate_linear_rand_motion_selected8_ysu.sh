#!/usr/bin/env bash
# Regenerate motion-blur dataset for 10 categories x 8 sequences (from raw CO3D *_selected_8.json).
# Blur: random angle in [0,360), kernel length 31, deterministic seed per sequence (base 123).
#
# BEFORE first run (destructive): empty output tree, e.g.
#   rm -rf /mnt/weka/hminasyan/outputs/degraded_frames_motion_10cat/*
#
# Submit:
#   cd ~/project_Hayk_Minasyan
#   sbatch scripts/motion_blur/run_generate_linear_rand_motion_selected8_ysu.sh

#SBATCH --job-name=motion_linear_rand_10cat8
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/motion_linear_rand_10cat8_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/motion_linear_rand_10cat8_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
RAW_CO3D="${RAW_CO3D:-/mnt/weka/hminasyan/data/co3d}"
MOTION_ROOT="${MOTION_ROOT:-/mnt/weka/hminasyan/outputs/degraded_frames_motion_10cat}"
MOTION_TAG="${MOTION_TAG:-linear_rand_angle_0_360_L31_seed123}"
KERNEL_LEN="${KERNEL_LEN:-31}"
SEED="${SEED:-123}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
fi

GEN="${PROJECT_DIR}/scripts/motion_blur/generate_linear_rand_angle_selected8.py"
[[ -f "${GEN}" ]] || { echo "ERROR: missing ${GEN}"; exit 1; }
[[ -d "${RAW_CO3D}" ]] || { echo "ERROR: missing RAW_CO3D=${RAW_CO3D}"; exit 1; }

mkdir -p "${MOTION_ROOT}"

echo "================================================================"
echo "Linear rand-angle motion blur | $(date) | $(hostname)"
echo "RAW_CO3D    : ${RAW_CO3D}"
echo "MOTION_ROOT : ${MOTION_ROOT}"
echo "MOTION_TAG  : ${MOTION_TAG}"
echo "KERNEL_LEN  : ${KERNEL_LEN}"
echo "SEED        : ${SEED}"
echo "================================================================"

export PYTHONUNBUFFERED=1
python -u "${GEN}" \
  --raw_co3d_root "${RAW_CO3D}" \
  --motion_root "${MOTION_ROOT}" \
  --motion_tag "${MOTION_TAG}" \
  --kernel_len "${KERNEL_LEN}" \
  --seed "${SEED}"

echo "Done: $(date)"
