#!/usr/bin/env bash
#SBATCH --job-name=deblurdinat_ft
# Slurm copies this script to /var/spool/slurmd/job... — BASH_SOURCE is not the repo.
# This makes the job start in the project root so PYTHONPATH and imports work.
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/deblurdinat_ft_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/deblurdinat_ft_%j.err
#
# Fine-tune DeblurDiNAT + DUSt3R on blurred CO3D (4 x GPU, DDP).
#
# Usage:
#   sbatch ~/project_Hayk_Minasyan/finetune_blur/deblurdinat/train.sh
#   (cwd when you submit does not matter; #SBATCH --chdir sets the repo root)
# Override sigma:
#   BLUR_SIGMAS="5" sbatch --export=ALL ~/project_Hayk_Minasyan/finetune_blur/deblurdinat/train.sh
# Override repo root (if different user/path):
#   PROJECT_DIR=/path/to/project_Hayk_Minasyan sbatch --export=ALL finetune_blur/deblurdinat/train.sh

set -euo pipefail

# After Slurm applies --chdir, the cwd is the project root (unless overridden).
: "${PROJECT_DIR:=$(pwd)}"
SCRIPT_DIR="${PROJECT_DIR}/finetune_blur/deblurdinat"

if [ ! -f "${SCRIPT_DIR}/train.py" ]; then
  echo "ERROR: train.py not found under ${SCRIPT_DIR}"
  echo "  Edit #SBATCH --chdir in this script to your repo, or run:"
  echo "  PROJECT_DIR=/path/to/project_Hayk_Minasyan sbatch --export=ALL,PROJECT_DIR ..."
  exit 1
fi

# ---- Paths (edit these for your cluster) ----
: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d_processed_10cat}"
: "${BLUR_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames_10cat}"
: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
: "${DEBLURDINAT_REPO:=/home/hminasyan/project_Hayk_Minasyan/DeblurDiNAT}"
: "${DEBLURDINAT_WEIGHTS:=${DEBLURDINAT_REPO}/results/DeblurDiNATL/models/DeblurDiNATL.pth}"
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_blur_runs/deblurdinat_512}"
: "${PYTHON:=/mnt/weka/hminasyan/deblurdinat_env/bin/python3}"

# ---- Hyperparameters ----
: "${BLUR_SIGMAS:=5 10 20}"
: "${FREEZE:=deblurdinat_only}"
# Per-GPU batch: ViT-L + DeblurDiNAT @512 is very heavy; use 1 + AMP if you OOM.
: "${BATCH_SIZE:=1}"
: "${EPOCHS:=30}"
: "${NUM_WORKERS:=8}"
: "${LR:=3e-5}"
: "${WARMUP_EPOCHS:=5}"
: "${GRAD_CLIP:=1.0}"
: "${RESOLUTION:=512}"
: "${AMP:=1}"
: "${GRAD_CHECKPOINT:=1}"
NPROC="${NPROC:-4}"

cd "${PROJECT_DIR}"
mkdir -p /home/hminasyan/project_Hayk_Minasyan/logs "${OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"
# Reduce memory fragmentation; critical for large models + NATTEN
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "================================================================"
echo "DeblurDiNAT + DUSt3R Fine-tuning"
echo "================================================================"
echo "Started   : $(date)"
echo "Host      : $(hostname)"
echo "SLURM Job : ${SLURM_JOB_ID:-interactive}"
echo "CO3D_ROOT : ${CO3D_ROOT}"
echo "BLUR_ROOT : ${BLUR_ROOT}"
echo "DUSt3R    : ${DUST3R_CKPT}"
echo "DeblurDiNAT: ${DEBLURDINAT_WEIGHTS}"
echo "Resolution: ${RESOLUTION}"
echo "Sigmas    : ${BLUR_SIGMAS}"
echo "Freeze    : ${FREEZE}"
echo "Batch/GPU : ${BATCH_SIZE}"
echo "AMP       : ${AMP}"
echo "Grad ckpt : ${GRAD_CHECKPOINT}"
echo "LR        : ${LR}  (warmup ${WARMUP_EPOCHS} ep, grad_clip ${GRAD_CLIP})"
echo "GPUs      : ${NPROC}"
echo "Output    : ${OUTPUT_DIR}/sigma_<N>"
echo "================================================================"

for SIGMA in ${BLUR_SIGMAS}; do
  SIGMA_OUTPUT="${OUTPUT_DIR}/sigma_${SIGMA}"
  mkdir -p "${SIGMA_OUTPUT}"

  echo ""
  echo ">>> Training sigma=${SIGMA}  -->  ${SIGMA_OUTPUT}"

  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    "${SCRIPT_DIR}/train.py" \
    --co3d_root            "${CO3D_ROOT}" \
    --blur_root            "${BLUR_ROOT}" \
    --blur_sigma           "${SIGMA}" \
    --dust3r_ckpt          "${DUST3R_CKPT}" \
    --deblurdinat_repo     "${DEBLURDINAT_REPO}" \
    --deblurdinat_weights  "${DEBLURDINAT_WEIGHTS}" \
    --output_dir           "${SIGMA_OUTPUT}" \
    --freeze               "${FREEZE}" \
    --batch_size           "${BATCH_SIZE}" \
    --epochs               "${EPOCHS}" \
    --num_workers          "${NUM_WORKERS}" \
    --lr                   "${LR}" \
    --warmup_epochs        "${WARMUP_EPOCHS}" \
    --grad_clip            "${GRAD_CLIP}" \
    --resolution           "${RESOLUTION}" \
    --amp                  "${AMP}" \
    --grad_checkpoint      "${GRAD_CHECKPOINT}" \
    "$@"
done

echo ""
echo "All done: $(date)"
