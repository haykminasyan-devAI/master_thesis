#!/usr/bin/env bash
#SBATCH --job-name=dust3r_deblur_joint_5_10_20_30_asds
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/dust3r_deblur_joint_5_10_20_30_asds_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/dust3r_deblur_joint_5_10_20_30_asds_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/finetune_blur/deblurdinat"
mkdir -p "${PROJECT_DIR}/logs"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
VENV_PYTHON="$(command -v python3)"

# Ensure CUDA NATTEN is available.
"${VENV_PYTHON}" - <<'PY' || { echo "ERROR: NATTEN missing CUDA lib. Install wheel from https://whl.natten.org"; exit 1; }
from natten._libnatten import HAS_LIBNATTEN
import sys
sys.exit(0 if HAS_LIBNATTEN else 1)
PY

: "${CO3D_ROOT:=${PROJECT_DIR}/data/co3d_processed_10cat}"
: "${BLUR_ROOT:=${PROJECT_DIR}/outputs/degraded_frames_10cat}"
: "${DUST3R_CKPT:=${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
# Start from DeblurDiNAT pretrained weights + DUSt3R checkpoint (no previous blur finetune resume).
: "${DEBLURDINAT_WEIGHTS:=${DEBLURDINAT_REPO}/results/DeblurDiNATL/models/DeblurDiNATL.pth}"
: "${OUTPUT_DIR:=${PROJECT_DIR}/finetune_blur_runs/deblurdinat_dust3r_asds_224_joint_5_10_20_30_from_dust3r}"

: "${BLUR_SIGMAS:=5 10 20 30}"
: "${FREEZE:=deblurdinat_only}"
: "${BATCH_SIZE:=1}"
: "${EPOCHS:=30}"
: "${LR:=2e-4}"
: "${WARMUP_EPOCHS:=0}"
: "${ETA_MIN:=1e-7}"
: "${GRAD_CLIP:=1.0}"
: "${WEIGHT_DECAY:=0.05}"
: "${RESOLUTION:=224}"
: "${AMP:=1}"
: "${GRAD_CHECKPOINT:=1}"
: "${NUM_WORKERS:=8}"
NPROC="${NPROC:-4}"

: "${WANDB_PROJECT:=master thesis}"
: "${WANDB_RUN_NAME:=deblurdinat_joint_5_10_20_30_from_dust3r}"

if [[ ! -f "${DUST3R_CKPT}" ]]; then
  echo "ERROR: DUSt3R checkpoint not found: ${DUST3R_CKPT}"
  exit 1
fi
if [[ ! -f "${DEBLURDINAT_WEIGHTS}" ]]; then
  echo "ERROR: DeblurDiNAT weights not found: ${DEBLURDINAT_WEIGHTS}"
  exit 1
fi
if [[ ! -d "${DEBLURDINAT_REPO}" ]]; then
  echo "ERROR: DeblurDiNAT repo not found: ${DEBLURDINAT_REPO}"
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

echo "================================================================"
echo "ASDS — DeblurDiNAT joint sigma(5,10,20,30) from DUSt3R ckpt | $(date) | $(hostname)"
echo "CO3D_ROOT        : ${CO3D_ROOT}"
echo "BLUR_ROOT        : ${BLUR_ROOT}"
echo "DUSt3R ckpt      : ${DUST3R_CKPT}"
echo "DeblurDiNAT init : ${DEBLURDINAT_WEIGHTS}"
echo "Output           : ${OUTPUT_DIR}"
echo "Sigmas           : ${BLUR_SIGMAS}"
echo "================================================================"

JOINT_OUTPUT="${OUTPUT_DIR}/joint_sigmas_$(echo "${BLUR_SIGMAS}" | tr ' ' '_')"
mkdir -p "${JOINT_OUTPUT}"

"${VENV_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  "${SCRIPT_DIR}/train.py" \
  --co3d_root            "${CO3D_ROOT}" \
  --blur_root            "${BLUR_ROOT}" \
  --blur_sigmas          ${BLUR_SIGMAS} \
  --dust3r_ckpt          "${DUST3R_CKPT}" \
  --deblurdinat_repo     "${DEBLURDINAT_REPO}" \
  --deblurdinat_weights  "${DEBLURDINAT_WEIGHTS}" \
  --output_dir           "${JOINT_OUTPUT}" \
  --freeze               "${FREEZE}" \
  --batch_size           "${BATCH_SIZE}" \
  --epochs               "${EPOCHS}" \
  --num_workers          "${NUM_WORKERS}" \
  --lr                   "${LR}" \
  --weight_decay         "${WEIGHT_DECAY}" \
  --warmup_epochs        "${WARMUP_EPOCHS}" \
  --eta_min              "${ETA_MIN}" \
  --grad_clip            "${GRAD_CLIP}" \
  --resolution           "${RESOLUTION}" \
  --amp                  "${AMP}" \
  --grad_checkpoint      "${GRAD_CHECKPOINT}" \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"}

echo ""
echo "All done: $(date)"
