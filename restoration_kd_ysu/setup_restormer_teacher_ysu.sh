#!/usr/bin/env bash
set -euo pipefail

# Run on YSU once before KD training.
# This script installs minimal deps in co3d_env and downloads frozen Restormer teacher checkpoints.

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
WEKA_HOME="${WEKA_HOME:-/mnt/weka/hminasyan}"
REST_DIR="${REST_DIR:-${PROJECT_DIR}/external/Restormer}"

source "${WEKA_HOME}/co3d_env/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p external
if [[ ! -d "${REST_DIR}" ]]; then
  git clone https://github.com/swz30/Restormer "${REST_DIR}"
fi

python -m pip install --upgrade pip
python -m pip install natsort scikit-image yacs addict future lmdb requests pyyaml einops

mkdir -p "${REST_DIR}/Motion_Deblurring/pretrained_models"
mkdir -p "${REST_DIR}/Defocus_Deblurring/pretrained_models"

if [[ ! -f "${REST_DIR}/Motion_Deblurring/pretrained_models/motion_deblurring.pth" ]]; then
  curl -L -o "${REST_DIR}/Motion_Deblurring/pretrained_models/motion_deblurring.pth" \
    "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth"
fi

if [[ ! -f "${REST_DIR}/Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth" ]]; then
  curl -L -o "${REST_DIR}/Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth" \
    "https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth"
fi

echo "Setup complete."
echo "Restormer root: ${REST_DIR}"
