#!/usr/bin/env bash
# ASDS (dgx): IFAN + DUSt3R defocus finetune — local data/checkpoint defaults (no Weka paths).
#
# Prereqs under PROJECT_DIR:
#   data/co3d_processed_10cat8seq_fixed/
#   data/co3d_selected_jsons/*_selected_8.json
#   checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
#   IFAN/ckpt/IFAN.pytorch
#
# Usage:
#   cd ~/project_Hayk_Minasyan
#   sbatch finetune_defocus/train_ifan_front_dust3r_defocus_asds.sh

#SBATCH -J ifan_dust3r_defocus_asds
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o logs/ifan_dust3r_defocus_asds_%j.log
#SBATCH -e logs/ifan_dust3r_defocus_asds_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
RAW_CO3D_ROOT="${RAW_CO3D_ROOT:-${PROJECT_DIR}/data/co3d_selected_jsons}"
CO3D_ROOT="${CO3D_ROOT:-${PROJECT_DIR}/data/co3d_processed_10cat8seq_fixed}"

_DUST3R_BASE="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
_WEFKA_DUST="/mnt/weka/hminasyan/checkpoints/${_DUST3R_BASE}"
if [[ -z "${DUST3R_CKPT:-}" ]]; then
  if [[ -f "${PROJECT_DIR}/checkpoints/${_DUST3R_BASE}" ]]; then
    DUST3R_CKPT="${PROJECT_DIR}/checkpoints/${_DUST3R_BASE}"
  else
    DUST3R_CKPT="${_WEFKA_DUST}"
  fi
fi
IFAN_REPO="${IFAN_REPO:-${PROJECT_DIR}/IFAN}"
IFAN_CKPT="${IFAN_CKPT:-${PROJECT_DIR}/IFAN/ckpt/IFAN.pytorch}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/finetune_defocus_runs/ifan_front_dust3r_seq8_randblur_asds}"

SEED="${SEED:-42}"
NPROC="${NPROC:-1}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-5e-5}"
RESOLUTION="${RESOLUTION:-224}"
DEFOCUS_RADIUS="${DEFOCUS_RADIUS:-6}"
DEFOCUS_TRAIN_MIN="${DEFOCUS_TRAIN_MIN:-3}"
DEFOCUS_TRAIN_MAX="${DEFOCUS_TRAIN_MAX:-9}"
WANDB_PROJECT="${WANDB_PROJECT:-dust3r-ifan-defocus}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ifan_defocus_seq8_asds_randblur_bs4_lr5e5}"
FINETUNE_FROM="${FINETUNE_FROM:-}"

CATEGORIES="${CATEGORIES:-apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot}"

VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/weka/hminasyan/co3d_env/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-co3d_env}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
  VENV_BIN="$(cd "$(dirname "${VENV_ACTIVATE}")" && pwd)"
  export PATH="${VENV_BIN}:${PATH}"
elif [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME}"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "ERROR: cannot activate Python env (venv, CONDA_SH, or ~/miniforge3)." >&2
  exit 1
fi

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python3 not on PATH after env activation" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import torch" 2>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} cannot import torch" >&2
  exit 1
fi

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

for f in "${DUST3R_CKPT}" "${IFAN_CKPT}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 1; }
done

"${PYTHON_BIN}" finetune_defocus/prepare_seqlevel_split_10cat8.py \
  --processed_root "${CO3D_ROOT}" \
  --raw_root "${RAW_CO3D_ROOT}" \
  --categories ${CATEGORIES} \
  --seed "${SEED}" \
  --n_train 6 --n_val 1 --n_test 1 \
  --train_name train_10cat8 \
  --val_name val_10cat8 \
  --test_name test_10cat8

DEFOCUS_TRAIN_FLAGS=()
if [[ -n "${DEFOCUS_TRAIN_MIN:-}" && -n "${DEFOCUS_TRAIN_MAX:-}" ]]; then
  DEFOCUS_TRAIN_FLAGS+=(--defocus_train_radius_min "${DEFOCUS_TRAIN_MIN}" --defocus_train_radius_max "${DEFOCUS_TRAIN_MAX}")
fi

FINETUNE_FLAGS=()
if [[ -n "${FINETUNE_FROM:-}" ]]; then
  if [[ ! -f "${FINETUNE_FROM}" ]]; then
    echo "ERROR: FINETUNE_FROM is set but not a file: ${FINETUNE_FROM}" >&2
    exit 1
  fi
  FINETUNE_FLAGS+=(--finetune_from_ckpt "${FINETUNE_FROM}")
fi

MASTER_PORT="${MASTER_PORT:-29621}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  finetune_defocus/train_ifan_front_dust3r_defocus.py \
  --co3d_root "${CO3D_ROOT}" \
  --categories ${CATEGORIES} \
  --defocus_radius "${DEFOCUS_RADIUS}" \
  "${DEFOCUS_TRAIN_FLAGS[@]}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --ifan_repo "${IFAN_REPO}" \
  --ifan_ckpt "${IFAN_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --freeze ifan_only \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers 16 \
  --lr "${LR}" \
  --eta_min 1e-6 \
  --weight_decay 0.01 \
  --warmup_epochs 3 \
  --grad_clip 1.0 \
  --resolution "${RESOLUTION}" \
  --amp 1 \
  --split_strategy predefined \
  --train_split train_10cat8 \
  --val_split val_10cat8 \
  --test_split test_10cat8 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --seed "${SEED}" \
  "${FINETUNE_FLAGS[@]}"

echo "Done: $(date)"
