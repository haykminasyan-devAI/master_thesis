#!/usr/bin/env bash
#SBATCH --job-name=kd_restormer_front
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/kd_restormer_1gpu_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/kd_restormer_1gpu_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
WEKA_HOME="${WEKA_HOME:-/mnt/weka/hminasyan}"

source "${WEKA_HOME}/co3d_env/bin/activate"
cd "${PROJECT_DIR}"
mkdir -p "${WEKA_HOME}/logs"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
RESTORMER_ROOT="${RESTORMER_ROOT:-${PROJECT_DIR}/external/Restormer}"
OUT_DIR="${OUTPUT_DIR:-/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_restormer_frontend_1gpu}"

# Total epochs (cosine T_max). Resume with RESUME=/path/to/student_last.pth and EPOCHS=50.
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"   # per GPU
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-3e-4}"
LR_MIN="${LR_MIN:-1e-6}"
WD="${WEIGHT_DECAY:-1e-4}"
NPROC="${NPROC:-1}"
RESUME="${RESUME:-}"
MOTION_K="${MOTION_KERNEL_SIZE:-35}"
DEFOCUS_R="${DEFOCUS_RADIUS:-9}"

EXTRA=()
if [[ -n "${RESUME}" ]]; then
  EXTRA+=( --resume "${RESUME}" )
fi

torchrun --nproc_per_node="${NPROC}" restoration_kd_ysu/train_kd_restormer_frontend.py \
  --data_root "${DATA_ROOT}" \
  --restormer_root "${RESTORMER_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --split_train "${SPLIT_TRAIN:-train_10cat8}" \
  --split_val "${SPLIT_VAL:-val_10cat8}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --lr_min "${LR_MIN}" \
  --weight_decay "${WD}" \
  --motion_kernel_size "${MOTION_K}" \
  --defocus_radius "${DEFOCUS_R}" \
  "${EXTRA[@]}"
