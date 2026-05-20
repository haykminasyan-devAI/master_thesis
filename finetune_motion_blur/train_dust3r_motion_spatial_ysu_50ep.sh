#!/usr/bin/env bash
#SBATCH --job-name=dust3r_deeprft_motion_50ep
#SBATCH --chdir=/home/hminasyan/project_Hayk_Minasyan
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_deeprft_motion_50ep_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/dust3r_deeprft_motion_50ep_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
else
  echo "ERROR: missing /mnt/weka/hminasyan/co3d_env/bin/activate"
  exit 1
fi
VENV_PYTHON="$(command -v python3)"

: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d}"
: "${MOTION_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames_motion_10cat}"
: "${MOTION_TAG:=spatial_patchwise_g8_k25_61_seed123}"

: "${DUST3R_CKPT:=/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${DEEPRFT_REPO:=${PROJECT_DIR}/DeepRFT}"
: "${DEEPRFT_CKPT:=/mnt/weka/hminasyan/checkpoints/deeprft/model_best.pth}"
: "${OUTPUT_DIR:=/mnt/weka/hminasyan/finetune_motion_blur_runs/deeprft_dust3r_ysu_224_spatial_g8_k25_61_50ep}"
: "${FREEZE:=deeprft_only}"

: "${EPOCHS:=50}"
: "${BATCH_SIZE:=1}"
: "${NUM_WORKERS:=8}"
: "${NPROC:=4}"
: "${RESOLUTION:=224}"
: "${AMP:=1}"
: "${GRAD_CHECKPOINT:=1}"
: "${GRAD_CLIP:=1.0}"
: "${LR:=2e-4}"
: "${WARMUP_EPOCHS:=0}"
: "${ETA_MIN:=1e-6}"
: "${WEIGHT_DECAY:=0.02}"

: "${WANDB_PROJECT:=master thesis}"
: "${WANDB_RUN_NAME:=deeprft_dust3r_motion_spatial_g8_k25_61_ysu_50ep}"

for f in "${DUST3R_CKPT}" "${DEEPRFT_CKPT}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing file: $f"; exit 1; }
done

export PROJECT_DIR CO3D_ROOT MOTION_ROOT MOTION_TAG
"${VENV_PYTHON}" - <<'PY'
import json, os, sys
project_dir = os.environ["PROJECT_DIR"]
motion_root = os.environ["MOTION_ROOT"]
motion_tag = os.environ["MOTION_TAG"]
seq_json = os.path.join(project_dir, "finetune_blur", "sequences_10cat.json")
seqs = json.load(open(seq_json))
missing = []
for c, sid in sorted(seqs.items()):
    d = os.path.join(motion_root, c, sid, motion_tag)
    if not os.path.isdir(d):
        missing.append(d)
if missing:
    print("ERROR: missing motion dirs:")
    for x in missing: print(" -", x)
    sys.exit(1)
print("Motion data sanity check passed.")
PY

export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

"${VENV_PYTHON}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  "${PROJECT_DIR}/finetune_motion_blur/train_deeprft_front_dust3r.py" \
  --co3d_root "${CO3D_ROOT}" \
  --motion_root "${MOTION_ROOT}" \
  --motion_tag "${MOTION_TAG}" \
  --dust3r_ckpt "${DUST3R_CKPT}" \
  --deeprft_repo "${DEEPRFT_REPO}" \
  --deeprft_weights "${DEEPRFT_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --freeze "${FREEZE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --eta_min "${ETA_MIN}" \
  --grad_clip "${GRAD_CLIP}" \
  --resolution "${RESOLUTION}" \
  --amp "${AMP}" \
  --grad_checkpoint "${GRAD_CHECKPOINT}" \
  --frontend_checkpoint 0 \
  --deeprft_num_res 8 \
  ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
  ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"}

echo "Done: $(date)"
