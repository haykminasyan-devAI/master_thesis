#!/usr/bin/env bash
# LoRA motion-blur training on a GPU node (do not run torchrun on the login node).
#
# Submit (4 GPU, defaults below):
#   cd ~/project_Hayk_Minasyan && sbatch motion_blur_ysu/submit_train_lora_research.slurm.sh
# Submit 1 GPU (override Slurm + NPROC; cancel any pending 4-GPU job first if replacing):
#   sbatch --job-name=lora_motion_1gpu --gres=gpu:1 --cpus-per-task=16 --mem=128G \
#     --output=/home/hminasyan/project_Hayk_Minasyan/logs/lora_motion_blur_1gpu_%j.log \
#     --error=/home/hminasyan/project_Hayk_Minasyan/logs/lora_motion_blur_1gpu_%j.err \
#     --export=ALL,NPROC=1,OUTPUT_DIR=/mnt/weka/hminasyan/finetune_motion_blur_runs/lora_motion_1gpu \
#     motion_blur_ysu/submit_train_lora_research.slurm.sh
# Override paths only:
#   sbatch --export=ALL,OUTPUT_DIR=/mnt/weka/hminasyan/finetune_motion_blur_runs/my_run motion_blur_ysu/submit_train_lora_research.slurm.sh
#
#SBATCH --job-name=lora_motion_4gpu
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/lora_motion_blur_4gpu_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/lora_motion_blur_4gpu_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ -f motion_blur_ysu/models/lora_cross_attn.py ]]; then
  echo "Removing stale motion_blur_ysu/models/ (conflicts with CroCo)."
  rm -rf motion_blur_ysu/models
fi

if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
else
  echo "ERROR: missing /mnt/weka/hminasyan/co3d_env/bin/activate"
  exit 1
fi

: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_motion_blur_runs/lora_motion_4gpu}"
: "${SPLIT_TRAIN:=train_10cat8}"
: "${SPLIT_VAL:=val_10cat8}"
: "${NPROC:=4}"
: "${BATCH_SIZE:=2}"
: "${EPOCHS:=10}"
: "${LR:=5e-5}"
: "${LORA_R:=8}"
: "${LORA_ALPHA:=8}"
: "${LORA_DROPOUT:=0.1}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

if ! python3 -c "import torch; n=${NPROC}; assert torch.cuda.is_available() and torch.cuda.device_count() >= n" 2>/dev/null; then
  echo "ERROR: this job needs at least ${NPROC} visible GPU(s). Check #SBATCH --gres and partition."
  python3 -c "import torch; print('cuda_available', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
  exit 1
fi

exec torchrun --nproc_per_node="${NPROC}" motion_blur_ysu/train_lora.py \
  --co3d_root "${CO3D_ROOT}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --split_train "${SPLIT_TRAIN}" \
  --split_val "${SPLIT_VAL}" \
  --lr "${LR}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}"
