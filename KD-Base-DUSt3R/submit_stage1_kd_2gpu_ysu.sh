#!/usr/bin/env bash
#SBATCH --job-name=kd_base_dust3r
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/kd_base_dust3r_2gpu_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/kd_base_dust3r_2gpu_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hminasyan/project_Hayk_Minasyan}"
SCRIPT="${PROJECT_DIR}/KD-Base-DUSt3R/train_stage1_encoder_kd.py"
PYTHON_BIN="${PYTHON_BIN:-/mnt/weka/hminasyan/co3d_env/bin/python3}"

DATA_ROOT="${DATA_ROOT:-/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
TEACHER_CKPT="${TEACHER_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTBase_BaseDecoder_224_linear.pth}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/weka/hminasyan/finetune_motion_blur_runs/kd_base_dust3r_stage1_2gpu}"
SPLIT_TRAIN="${SPLIT_TRAIN:-train_10cat8}"
SPLIT_VAL="${SPLIT_VAL:-val_10cat8}"
STRIDE="${STRIDE:-5}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"   # per GPU
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
LR_MIN="${LR_MIN:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-master-thesis}"
WANDB_ENTITY="${WANDB_ENTITY:-hayk-minasyan-yerevan-state-university-ysu}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-kd_base_dust3r_stage1}"

NPROC="${NPROC:-2}"

mkdir -p /mnt/weka/hminasyan/logs "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME

echo "================================================================"
echo "Stage-1 KD: Base DUSt3R encoder -> DUSt3R encoder LoRA"
echo "================================================================"
echo "Started      : $(date)"
echo "Host         : $(hostname)"
echo "Data root    : ${DATA_ROOT}"
echo "Teacher ckpt    : ${TEACHER_CKPT}"
echo "DUSt3R ckpt  : ${DUST3R_CKPT}"
echo "Split train  : ${SPLIT_TRAIN}"
echo "Split val    : ${SPLIT_VAL}"
echo "Stride       : ${STRIDE}"
echo "Image size   : ${IMAGE_SIZE}"
echo "Epochs       : ${EPOCHS}"
echo "Batch/GPU    : ${BATCH_SIZE}"
echo "LR           : ${LR} -> ${LR_MIN}"
echo "Weight decay : ${WEIGHT_DECAY}"
echo "LoRA r/alpha : ${LORA_R}/${LORA_ALPHA}"
echo "GPUs         : ${NPROC}"
echo "Output       : ${OUTPUT_DIR}"
echo "W&B project  : ${WANDB_PROJECT}"
echo "W&B entity   : ${WANDB_ENTITY}"
echo "W&B run name : ${WANDB_RUN_NAME}"
echo "================================================================"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  "${SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --split_train "${SPLIT_TRAIN}" \
  --split_val "${SPLIT_VAL}" \
  --stride "${STRIDE}" \
  --image_size "${IMAGE_SIZE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --lr_min "${LR_MIN}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --seed "${SEED}" \
  --log_interval "${LOG_INTERVAL}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_run_name "${WANDB_RUN_NAME}"

echo "Done: $(date)"
