#!/usr/bin/env bash
# YSU: IFAN(front) + DUSt3R defocus finetune with sequence-level 10cat x 8seq split.
#
# Usage on YSU:
#   cd ~/project_Hayk_Minasyan
#   sbatch finetune_defocus/train_ifan_front_dust3r_defocus_ysu_seq8.sh
#
# Optional overrides:
#   sbatch --export=ALL,SEED=7,EPOCHS=50,OUT_DIR=/path/to/run finetune_defocus/train_ifan_front_dust3r_defocus_ysu_seq8.sh
#
# Train blur: default random integer in [DEFOCUS_TRAIN_MIN, DEFOCUS_TRAIN_MAX] (val/test use DEFOCUS_RADIUS).
# Fixed train blur = same as val: unset both, e.g.  sbatch --export=ALL,DEFOCUS_TRAIN_MIN=,DEFOCUS_TRAIN_MAX=,...
#
# Warm start from a previous run's best (new OUT_DIR, epoch 0, fresh optimizer):
#   sbatch --export=ALL,FINETUNE_FROM=${HOME}/project_Hayk_Minasyan/finetune_defocus_runs/ifan_front_dust3r_seq8_ysu/checkpoint-best-val.pth,...
#
# After training, use checkpoint-best-val.pth (lowest val loss; best_val_epoch stored in the file).
#
# Defaults: YSU partition=research, 4× GPU DDP (--gres=gpu:4, not gpu:a100:4), batch_size=2 per GPU.
# If submit fails, lower --mem/--cpus or ask: sinfo -p research -o "%P %G %c %m".
# 1-GPU research: sbatch --export=ALL,NPROC=1,BATCH_SIZE=4 --gres=gpu:1 (edit #SBATCH gres or use a 1-GPU wrapper script).

#SBATCH -J ifan_dust3r_defocus_seq8_4gpu
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH -o logs/ifan_dust3r_defocus_seq8_4gpu_%j.log
#SBATCH -e logs/ifan_dust3r_defocus_seq8_4gpu_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
RAW_CO3D_ROOT="${RAW_CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d}"
CO3D_ROOT="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
# 512 ckpt name is common on clusters even when training at 224 resolution.
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
IFAN_REPO="${IFAN_REPO:-${PROJECT_DIR}/IFAN}"
IFAN_CKPT="${IFAN_CKPT:-${PROJECT_DIR}/IFAN/ckpt/IFAN.pytorch}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/weka/hminasyan/runs/finetune_defocus_runs}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/ifan_front_dust3r_seq8_randblur_ysu_4gpu}"

SEED="${SEED:-42}"
NPROC="${NPROC:-4}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"
RESOLUTION="${RESOLUTION:-224}"
# Val/test always use this radius; train uses random in [DEFOCUS_TRAIN_MIN, DEFOCUS_TRAIN_MAX] if both set.
DEFOCUS_RADIUS="${DEFOCUS_RADIUS:-6}"
DEFOCUS_TRAIN_MIN="${DEFOCUS_TRAIN_MIN:-3}"
DEFOCUS_TRAIN_MAX="${DEFOCUS_TRAIN_MAX:-9}"
WANDB_PROJECT="${WANDB_PROJECT:-dust3r-ifan-defocus}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ifan_defocus_seq8_4gpu_randblur_bs2per_gpu_lr5e5}"
# Optional: path to prior checkpoint-best-val.pth (or any .pth with "model" key). Ignored if checkpoint-last exists in OUT_DIR.
FINETUNE_FROM="${FINETUNE_FROM:-}"

CATEGORIES="${CATEGORIES:-apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot}"

# Do not `source ~/.bashrc` in batch: cluster dotfiles often run sudo/module hooks and kill jobs early.
# Set CONDA_SH if conda lives elsewhere, e.g. sbatch --export=ALL,CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
# YSU: co3d_env is often a venv; ASDS: conda env — use CONDA_SH or default Miniforge.
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/weka/hminasyan/co3d_env/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-co3d_env}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
  # Slurm batch PATH may omit the venv bin; torchrun is a small shim — prefer explicit PATH + python -m.
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
  echo "ERROR: cannot activate Python env. Set VENV_ACTIVATE, CONDA_SH, or install Miniforge at ~/miniforge3" >&2
  exit 1
fi

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python3 not on PATH after env activation (PATH=${PATH})" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import torch" 2>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} cannot import torch; fix co3d_env or VENV_ACTIVATE" >&2
  exit 1
fi

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" finetune_defocus/prepare_seqlevel_split_10cat8.py \
  --processed_root "${CO3D_ROOT}" \
  --raw_root "${RAW_CO3D_ROOT}" \
  --categories ${CATEGORIES} \
  --seed "${SEED}" \
  --n_train 6 --n_val 1 --n_test 1 \
  --train_name train_10cat8 \
  --val_name val_10cat8 \
  --test_name test_10cat8

# Fixed train blur: export DEFOCUS_TRAIN_MIN= DEFOCUS_TRAIN_MAX= (empty) before sbatch to omit these flags.
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

MASTER_PORT="${MASTER_PORT:-29622}"
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
  --num_workers "${NUM_WORKERS}" \
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

