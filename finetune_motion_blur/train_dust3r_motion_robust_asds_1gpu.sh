#!/usr/bin/env bash
#SBATCH -J dust3r_motion_robust
#SBATCH -p a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o logs/dust3r_motion_robust_asds_%j.log
#SBATCH -e logs/dust3r_motion_robust_asds_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
CO3D_ROOT="${CO3D_ROOT:-${PROJECT_DIR}/data/co3d_processed_10cat}"
DUST3R_CKPT="${DUST3R_CKPT:-${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/finetune_motion_blur_runs/dust3r_motion_robust_asds_1gpu_${SLURM_JOB_ID}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-co3d_env}"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

python -m torch.distributed.run --nproc_per_node="${NPROC:-4}" \
  finetune_motion_blur/train_dust3r_motion_robust.py \
  --co3d_root "${CO3D_ROOT}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --epochs "${EPOCHS:-15}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --resolution "${RESOLUTION:-224}" \
  --lr_main "${LR_MAIN:-1e-5}" \
  --lr_encoder_low "${LR_ENCODER_LOW:-1e-6}" \
  --encoder_blocks_train "${ENCODER_BLOCKS_TRAIN:-4}" \
  --unfreeze_full_encoder_epoch "${UNFREEZE_EPOCH:--1}" \
  --kernel_min "${KERNEL_MIN:-3}" \
  --kernel_max "${KERNEL_MAX:-9}" \
  --train_clean_only "${TRAIN_CLEAN_ONLY:-0}" \
  --amp "${AMP:-1}" \
  --print_freq "${PRINT_FREQ:-20}" \
  --seed "${SEED:-42}"
