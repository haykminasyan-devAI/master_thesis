#!/usr/bin/env bash
#SBATCH --job-name=eval_ifan_defocus
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/eval_ifan_defocus_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/eval_ifan_defocus_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
CO3D_ROOT="${CO3D_ROOT:-${PROJECT_DIR}/data/co3d_processed_10cat}"
DUST3R_CKPT="${DUST3R_CKPT:-${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
IFAN_REPO="${IFAN_REPO:-${PROJECT_DIR}/IFAN}"
IFAN_CKPT="${IFAN_CKPT:-${PROJECT_DIR}/IFAN/ckpt/IFAN.pytorch}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${PROJECT_DIR}/finetune_defocus_runs/ifan_front_dust3r_freeze_train10_asds}"
FINETUNED_CKPT="${FINETUNED_CKPT:-${TRAIN_OUT_DIR}/checkpoint-best-val.pth}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${TRAIN_OUT_DIR}/eval_same6}"
EVAL_CATEGORIES="${EVAL_CATEGORIES:-bottle cup donut teddybear couch toytrain}"
DEFOCUS_RADIUS="${DEFOCUS_RADIUS:-6}"
RESOLUTION="${RESOLUTION:-224}"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
PYTHON="$(command -v python3)"

mkdir -p "${PROJECT_DIR}/logs" "${EVAL_OUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

"${PYTHON}" "${PROJECT_DIR}/finetune_defocus/eval_ifan_front_dust3r_defocus.py" \
  --co3d_root "${CO3D_ROOT}" \
  --categories ${EVAL_CATEGORIES} \
  --defocus_radius "${DEFOCUS_RADIUS}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --ifan_repo "${IFAN_REPO}" \
  --ifan_ckpt "${IFAN_CKPT}" \
  --finetuned_ckpt "${FINETUNED_CKPT}" \
  --resolution "${RESOLUTION}" \
  --batch_size 1 \
  --num_workers 4 \
  --amp 1 \
  --output_dir "${EVAL_OUT_DIR}"
