#!/usr/bin/env bash
#SBATCH --job-name=kd_zdce_uretinex
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/kd_zerodce_uretinex_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/kd_zerodce_uretinex_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
WEKA_HOME="${WEKA_HOME:-/mnt/weka/hminasyan}"

# shellcheck source=/dev/null
source "${WEKA_HOME}/co3d_env/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p "${WEKA_HOME}/logs"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

CO3D_ROOT="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
OUT_DIR="${OUTPUT_DIR:-${WEKA_HOME}/finetune_motion_blur_runs/kd_zerodce_uretinex}"
URETINEX_ROOT="${URETINEX_ROOT:-${PROJECT_DIR}/external/URetinex-Net}"
ZERODCE_ROOT="${ZERODCE_ROOT:-${PROJECT_DIR}/external/Zero-DCE}"

export SPLIT_TRAIN="${SPLIT_TRAIN:-train_10cat8_7v1}"
export SPLIT_VAL="${SPLIT_VAL:-val_10cat8_7v1}"

export BATCH_SIZE="${BATCH_SIZE:-16}"
export EPOCHS="${EPOCHS:-300}"
export LR="${LR:-1e-4}"
export WD="${WEIGHT_DECAY:-1e-5}"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  "${PROJECT_DIR}/KD-Zero-Reference/train_kd_zerodce.py" \
  --co3d_root "${CO3D_ROOT}" \
  --split_train "${SPLIT_TRAIN}" \
  --split_val "${SPLIT_VAL}" \
  --uretinex_root "${URETINEX_ROOT}" \
  --zerodce_root "${ZERODCE_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WD}" \
  "$@"

echo "Done. Outputs: ${OUT_DIR}"
