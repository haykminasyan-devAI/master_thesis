#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash inference_rest/run_restormer_inference.sh
# Optional env overrides:
#   PROJECT_DIR=~/project_Hayk_Minasyan
#   INPUT_ROOT=~/project_Hayk_Minasyan/inference_rest/corrupted/teddybear
#   RESULT_ROOT=~/project_Hayk_Minasyan/inference_rest/restored/teddybear
#   TILE=720

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
RESTORMER_DIR="${PROJECT_DIR}/inference_rest/Restormer"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_DIR}/inference_rest/corrupted/teddybear}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/inference_rest/restored/teddybear}"
TILE="${TILE:-}"
TILE_OVERLAP="${TILE_OVERLAP:-32}"

[[ -d "${RESTORMER_DIR}" ]] || { echo "Missing ${RESTORMER_DIR}"; exit 1; }
[[ -d "${INPUT_ROOT}" ]] || { echo "Missing ${INPUT_ROOT}"; exit 1; }
mkdir -p "${RESULT_ROOT}"

cd "${RESTORMER_DIR}"

# Map corruption folder -> Restormer task name
declare -A TASK_MAP=(
  ["noise"]="Gaussian_Color_Denoising"
  ["blur"]="Motion_Deblurring"
  ["motion_blur"]="Motion_Deblurring"
  ["raining"]="Deraining"
  ["defocus_blur"]="Single_Image_Defocus_Deblurring"
)

for corr in noise blur motion_blur raining defocus_blur; do
  in_dir="${INPUT_ROOT}/${corr}"
  [[ -d "${in_dir}" ]] || { echo "Skip ${corr} (missing ${in_dir})"; continue; }
  task="${TASK_MAP[$corr]}"
  echo "=== ${corr} -> ${task} ==="
  if [[ -n "${TILE}" ]]; then
    python demo.py --task "${task}" --input_dir "${in_dir}" --result_dir "${RESULT_ROOT}" --tile "${TILE}" --tile_overlap "${TILE_OVERLAP}"
  else
    python demo.py --task "${task}" --input_dir "${in_dir}" --result_dir "${RESULT_ROOT}"
  fi
done

echo "Restored outputs saved under: ${RESULT_ROOT}"
