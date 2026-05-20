#!/usr/bin/env bash
# DeblurDiNAT + DUSt3R (motion/defocus) on 4 CO3D categories only (faster epochs).
#
# CO3D root must match selected_seqs_train_10cat8.json. For
# co3d_processed_10cat8seq_fixed the 10 categories are:
#   apple baseballbat baseballglove banana bicycle bowl broccoli cake carrot car
#
# Default below: apple banana bowl car (change with CATEGORIES="..." if you like).
#
# Submit:
#   sbatch "finetuning Motion&Defocus/deblurdinat/submit_train_motion_defocus_4cat_research.slurm.sh"
#
# Example override:
#   CATEGORIES="bicycle broccoli cake carrot" sbatch ...

#SBATCH --job-name=dinat_md_4cat
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/weka/hminasyan/logs/dinat_motion_defocus_4cat_%j.log
#SBATCH --error=/mnt/weka/hminasyan/logs/dinat_motion_defocus_4cat_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
: "${SCRIPT_DIR:=${PROJECT_DIR}/finetuning Motion&Defocus/deblurdinat}"

if [ ! -f "${SCRIPT_DIR}/train_motion_defocus.py" ]; then
  echo "ERROR: train_motion_defocus.py not found under ${SCRIPT_DIR}"
  exit 1
fi

# YSU paths
: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d_processed_10cat8seq_fixed}"
: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
: "${DEBLURDINAT_WEIGHTS:=${DEBLURDINAT_REPO}/results/DeblurDiNATL/models/DeblurDiNATL.pth}"
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_motion_blur_runs/deblurdinat_motion_defocus_512_4cat}"
: "${PYTHON:=/mnt/weka/hminasyan/deblurdinat_env/bin/python3}"

# Categories = top-level keys in selected_seqs_train_10cat8.json (folder names under CO3D_ROOT)
CAT_STR="${CATEGORIES:-apple banana bowl car}"
read -r -a CAT_ARR <<< "${CAT_STR}"

: "${FREEZE:=deblurdinat_only}"
: "${MOTION_PROB:=0.5}"
: "${BATCH_SIZE:=1}"
: "${EPOCHS:=30}"
: "${NUM_WORKERS:=8}"
: "${LR:=3e-5}"
: "${WARMUP_EPOCHS:=5}"
: "${ETA_MIN:=1e-7}"
: "${WEIGHT_DECAY:=0.05}"
: "${GRAD_CLIP:=1.0}"
: "${RESOLUTION:=512}"
: "${AMP:=1}"
: "${GRAD_CHECKPOINT:=1}"
: "${DEBLUR_CHECKPOINT:=1}"
: "${SPLIT_TRAIN:=train_10cat8}"
: "${VAL_RATIO:=0.20}"
: "${TEST_RATIO:=0.00}"
: "${WANDB_PROJECT:=master-thesis}"
: "${WANDB_RUN_NAME:=dinat_md_4cat}"
: "${WANDB_ENTITY:=hayk-minasyan-yerevan-state-university-ysu}"
NPROC="${NPROC:-2}"

cd "${PROJECT_DIR}"
mkdir -p /mnt/weka/hminasyan/logs "${OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export WANDB_ENTITY="${WANDB_ENTITY}"

echo "================================================================"
echo "DeblurDiNAT + DUSt3R — 4 categories only"
echo "================================================================"
echo "Started     : $(date)"
echo "Host        : $(hostname)"
echo "SLURM Job   : ${SLURM_JOB_ID:-interactive}"
echo "CO3D_ROOT   : ${CO3D_ROOT}"
echo "Categories  : ${CAT_ARR[*]}"
echo "DUSt3R ckpt : ${DUST3R_CKPT}"
echo "DeblurDiNAT : ${DEBLURDINAT_WEIGHTS}"
echo "Train split : ${SPLIT_TRAIN}"
echo "Output      : ${OUTPUT_DIR}"
echo "================================================================"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  "${SCRIPT_DIR}/train_motion_defocus.py" \
  --co3d_root            "${CO3D_ROOT}" \
  --dust3r_ckpt          "${DUST3R_CKPT}" \
  --deblurdinat_repo     "${DEBLURDINAT_REPO}" \
  --deblurdinat_weights  "${DEBLURDINAT_WEIGHTS}" \
  --output_dir           "${OUTPUT_DIR}" \
  --freeze               "${FREEZE}" \
  --motion_prob          "${MOTION_PROB}" \
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
  --deblur_checkpoint    "${DEBLUR_CHECKPOINT}" \
  --split_train          "${SPLIT_TRAIN}" \
  --val_ratio            "${VAL_RATIO}" \
  --test_ratio           "${TEST_RATIO}" \
  --wandb_project        "${WANDB_PROJECT}" \
  --wandb_run_name       "${WANDB_RUN_NAME}" \
  --categories           "${CAT_ARR[@]}" \
  "$@"

echo "Done: $(date)"
