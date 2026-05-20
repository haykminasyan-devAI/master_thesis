#!/usr/bin/env bash
# YSU: Uformer(front) + DUSt3R motion-blur finetune with sequence-level 10cat x 8seq split
# (same selected_seqs_* JSONs as defocus: prepare_seqlevel_split_10cat8.py).
#
# Split default: train/val/test sequences per category = 6 / 2 / 0 (no held-out test set in this job;
# evaluate later on separate data). Override counts with N_TRAIN, N_VAL, N_TEST if needed.
#
# YSU cluster portal: https://cluster.ysu.am/
#
# Defaults: 1x H100, per-GPU batch_size=2. Override NPROC / BATCH_SIZE to match multi-GPU sbatch --gres.
# Usage:
#   cd ~/project_Hayk_Minasyan
#   sbatch finetune_motion_blur/train_uformer_front_dust3r_motion_ysu_seq8.sh
#
# Requires: pip install 'peft>=0.11' (see dust3r/requirements.txt).
# Legacy full Uformer front (no LoRA): sbatch --export=ALL,USE_PEFT_LORA=0,COSINE_PER_ITERATION=0,LR=2e-4,...
# Motion blur style (subdir under MOTION_ROOT/<cat>/<seq>/):
#   Default: linear_rand_angle_0_360_L31_seed123 (matches YSU degraded_frames_motion_10cat layout).
# Overrides:
#   sbatch --export=ALL,MOTION_TAG=temporal_avg_w11_gopro_like,... \
#     finetune_motion_blur/train_uformer_front_dust3r_motion_ysu_seq8.sh

#SBATCH -J uformer_dust3r_motion_seq8
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 96:00:00
#SBATCH -o logs/uformer_dust3r_motion_seq8_%j.log
#SBATCH -e logs/uformer_dust3r_motion_seq8_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
RAW_CO3D_ROOT="${RAW_CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d}"
CO3D_ROOT="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
MOTION_ROOT="${MOTION_ROOT:-/mnt/weka/hminasyan/outputs/degraded_frames_motion_10cat}"
MOTION_TAG="${MOTION_TAG:-linear_rand_angle_0_360_L31_seed123}"

_DUST3R_BASE="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
_WEFKA_DUST="/mnt/weka/hminasyan/checkpoints/${_DUST3R_BASE}"
if [[ -z "${DUST3R_CKPT:-}" ]]; then
  if [[ -f "${PROJECT_DIR}/checkpoints/${_DUST3R_BASE}" ]]; then
    DUST3R_CKPT="${PROJECT_DIR}/checkpoints/${_DUST3R_BASE}"
  else
    DUST3R_CKPT="${_WEFKA_DUST}"
  fi
fi
UFORMER_REPO="${UFORMER_REPO:-${PROJECT_DIR}/Uformer}"
_UFORMER_BASE="Uformer_B.pth"
_WEFKA_UFORMER="/mnt/weka/hminasyan/checkpoints/uformer/${_UFORMER_BASE}"
if [[ -z "${UFORMER_WEIGHTS:-}" ]]; then
  if [[ -f "${PROJECT_DIR}/checkpoints/uformer/${_UFORMER_BASE}" ]]; then
    UFORMER_WEIGHTS="${PROJECT_DIR}/checkpoints/uformer/${_UFORMER_BASE}"
  elif [[ -f "${UFORMER_REPO}/pretrained_model/${_UFORMER_BASE}" ]]; then
    UFORMER_WEIGHTS="${UFORMER_REPO}/pretrained_model/${_UFORMER_BASE}"
  else
    UFORMER_WEIGHTS="${_WEFKA_UFORMER}"
  fi
fi
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/finetune_motion_blur_runs/uformer_dust3r_seq8_motion_ysu}"

TRAIN_NAME="${TRAIN_NAME:-train_10cat8}"
VAL_NAME="${VAL_NAME:-val_10cat8}"
TEST_NAME="${TEST_NAME:-test_10cat8}"

SEED="${SEED:-42}"
N_TRAIN="${N_TRAIN:-6}"
N_VAL="${N_VAL:-2}"
N_TEST="${N_TEST:-0}"
EPOCHS="${EPOCHS:-50}"
# Per-GPU batch size; with DDP, effective batch ≈ BATCH_SIZE * NPROC.
BATCH_SIZE="${BATCH_SIZE:-2}"
NPROC="${NPROC:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# LoRA motion defaults (see finetune_motion_blur/model_uformer_motion_lora.py)
LR="${LR:-2e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
ETA_MIN="${ETA_MIN:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
RESOLUTION="${RESOLUTION:-224}"
AMP="${AMP:-1}"
FREEZE="${FREEZE:-uformer_only}"
USE_PEFT_LORA="${USE_PEFT_LORA:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
COSINE_PER_ITERATION="${COSINE_PER_ITERATION:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-dust3r-uformer-motion}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-uformer_dust3r_motion_seq8_lora_1gpu_bs2_lr2e5}"

CATEGORIES="${CATEGORIES:-apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot}"

# YSU: co3d_env is often a Python venv; ASDS: conda env (no bin/activate) — use CONDA_SH or default Miniforge.
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/weka/hminasyan/co3d_env/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-co3d_env}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
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

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

for f in "${DUST3R_CKPT}" "${UFORMER_WEIGHTS}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 1; }
done

python finetune_defocus/prepare_seqlevel_split_10cat8.py \
  --processed_root "${CO3D_ROOT}" \
  --raw_root "${RAW_CO3D_ROOT}" \
  --categories ${CATEGORIES} \
  --seed "${SEED}" \
  --n_train "${N_TRAIN}" --n_val "${N_VAL}" --n_test "${N_TEST}" \
  --train_name "${TRAIN_NAME}" \
  --val_name "${VAL_NAME}" \
  --test_name "${TEST_NAME}"

export CO3D_ROOT MOTION_ROOT MOTION_TAG TRAIN_NAME VAL_NAME TEST_NAME
python3 <<'PY'
import json, os, sys

co3d_root = os.environ["CO3D_ROOT"]
motion_root = os.environ["MOTION_ROOT"]
motion_tag = os.environ["MOTION_TAG"]
names = [
    os.environ["TRAIN_NAME"],
    os.environ["VAL_NAME"],
    os.environ["TEST_NAME"],
]
missing = []
for name in names:
    jpath = os.path.join(co3d_root, f"selected_seqs_{name}.json")
    if not os.path.isfile(jpath):
        if name == os.environ.get("TEST_NAME", "test_10cat8"):
            print(f"WARNING: missing {jpath} (skipping motion check for this split)")
            continue
        print(f"ERROR: missing split json: {jpath}", file=sys.stderr)
        sys.exit(1)
    data = json.load(open(jpath))
    for cat, scenes in data.items():
        if not isinstance(scenes, dict):
            continue
        for sid in scenes:
            d = os.path.join(motion_root, cat, sid, motion_tag)
            if not os.path.isdir(d):
                missing.append(d)
if missing:
    print("ERROR: motion frames not found for:", file=sys.stderr)
    for x in sorted(set(missing))[:50]:
        print(" ", x, file=sys.stderr)
    if len(set(missing)) > 50:
        print(f"  ... and {len(set(missing)) - 50} more", file=sys.stderr)
    sys.exit(1)
print("Motion data layout sanity check passed.")
PY

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${UFORMER_REPO}:${PYTHONPATH:-}"

MASTER_PORT="${MASTER_PORT:-29631}"
torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  finetune_motion_blur/train_uformer_front_dust3r.py \
  --co3d_root "${CO3D_ROOT}" \
  --motion_root "${MOTION_ROOT}" \
  --motion_tag "${MOTION_TAG}" \
  --categories ${CATEGORIES} \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --uformer_repo "${UFORMER_REPO}" \
  --uformer_weights "${UFORMER_WEIGHTS}" \
  --output_dir "${OUT_DIR}" \
  --freeze "${FREEZE}" \
  --use_peft_lora "${USE_PEFT_LORA}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --cosine_per_iteration "${COSINE_PER_ITERATION}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --eta_min "${ETA_MIN}" \
  --grad_clip "${GRAD_CLIP}" \
  --resolution "${RESOLUTION}" \
  --amp "${AMP}" \
  --split_strategy predefined \
  --train_split "${TRAIN_NAME}" \
  --val_split "${VAL_NAME}" \
  --test_split "${TEST_NAME}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --seed "${SEED}"

echo "Done: $(date)"
