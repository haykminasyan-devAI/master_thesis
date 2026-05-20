#!/usr/bin/env bash
# 6 CO3D sequences — same category/seq list as blur+noise eval (source SEQUENCES_FILE).
#
# Motion-blurred inputs under:
#   MOTION_ROOT/<category>/<seq_id>/<MOTION_TAG>/frame*.jpg
# GT point clouds from:
#   CO3D_ROOT/<category>/<seq_id>/pointcloud.ply
#
# Pipelines: DUSt3R | Uformer (GoPro-pretrained) + DUSt3R | motion-finetuned Uformer + DUSt3R
# (Finetuned checkpoint is from finetune_motion_blur, same Uformer+DUSt3R wrapper as finetune_noise.)
#
# Prerequisite: for each of the 6 (category, seq_id) pairs, motion frames must exist.
# If you only generated motion for the 10-train-cat set, run your temporal-averaging script
# for these six sequences (or set MOTION_ROOT to a tree that already contains them).
#
# YSU weka (only has hydrant/toybus six on disk): often
#   export SEQUENCES_FILE=$PWD/evaluation_blur_and_noise/sequences_6eval_weka.inc.sh
#
# Usage:
#   cd /path/to/project_Hayk_Minasyan
#   sbatch -p research --export=ALL,PROJECT_DIR=...,MOTION_ROOT=...,MOTION_FINETUNED_CKPT=... \
#     evaluation_blur_and_noise/run_eval_6seq_motion_chamfer.sh
#
#SBATCH --job-name=eval6seq_motion
#   cd /path/to/project_Hayk_Minasyan && sbatch evaluation_blur_and_noise/run_eval_6seq_motion_chamfer.sh
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=eval_6seq_motion_%j.log
#SBATCH --error=eval_6seq_motion_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == /var/spool/slurmd/* ]]; then
  if [[ -n "${PROJECT_DIR:-}" && -d "${PROJECT_DIR}/dust3r" ]]; then
    :
  elif [[ -d "$(pwd)/dust3r" ]]; then
    PROJECT_DIR="$(pwd -P)"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/dust3r" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    echo "ERROR: Under sbatch, set PROJECT_DIR to repo root, or submit from repo with cd."
    exit 1
  fi
  EVAL_DIR="${PROJECT_DIR}/evaluation_blur_and_noise"
else
  : "${PROJECT_DIR:=$(cd "${SCRIPT_DIR}/.." && pwd)}"
  EVAL_DIR="${SCRIPT_DIR}"
fi
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
elif [[ -f /home/asds/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/asds/miniforge3/etc/profile.d/conda.sh
  conda activate co3d_env
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
fi
VENV_PYTHON="$(command -v python3)"
INF_MOTION="${EVAL_DIR}/uformer_run_inference.py"

# shellcheck source=/dev/null
# shellcheck disable=SC1091
source "${SEQUENCES_FILE:-${EVAL_DIR}/sequences_6eval.inc.sh}"

: "${N_FRAMES:=20}"
: "${IMAGE_SIZE:=224}"
: "${DUST3R_DIR:=${PROJECT_DIR}/dust3r}"
: "${DUST3R_CKPT:=${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${UFORMER_REPO:=${PROJECT_DIR}/Uformer}"
# GoPro deblurring weights (not SIDD) for a meaningful pretrained baseline on motion blur
: "${UFORMER_GOPRO:=/mnt/weka/hminasyan/checkpoints/uformer/Uformer_B.pth}"
if [[ ! -f "${UFORMER_GOPRO}" ]]; then
  if [[ -f "${PROJECT_DIR}/checkpoints/uformer/Uformer_B.pth" ]]; then
    UFORMER_GOPRO="${PROJECT_DIR}/checkpoints/uformer/Uformer_B.pth"
  else
    UFORMER_GOPRO="${UFORMER_REPO}/pretrained_model/Uformer_B.pth"
  fi
fi

: "${CO3D_ROOT:=${PROJECT_DIR}/data/co3d_processed_10cat}"
# Layout: MOTION_ROOT/<cat>/<seq_id>/<MOTION_TAG>/
: "${MOTION_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames_motion_6seq}"
: "${MOTION_TAG:=temporal_avg_w11_gopro_like}"
: "${MOTION_FINETUNED_CKPT:=${PROJECT_DIR}/finetune_motion_blur_runs/motion_dust3r/joint_run/checkpoint-best-val.pth}"
if [[ ! -f "${MOTION_FINETUNED_CKPT}" ]]; then
  _M="/mnt/weka/hminasyan/finetune_motion_blur_runs/uformer_dust3r_ysu_224_temporal_avg_w11_50ep/checkpoint-best-val.pth"
  if [[ -f "${_M}" ]]; then
    MOTION_FINETUNED_CKPT="${_M}"
  fi
fi
unset _M 2>/dev/null || true

: "${OUT_ROOT:=${PROJECT_DIR}/outputs/dust3r_eval_6seq_motion_$(date +%Y%m%d_%H%M%S)}"
: "${RESULT_CSV:=${OUT_ROOT}/chamfer_results_motion_6seq.csv}"

export PYTHONPATH="${PROJECT_DIR}/dust3r/croco:${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${UFORMER_REPO}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}"

echo "================================================================"
echo "6-seq motion Chamfer eval | $(date) | $(hostname)"
echo "SEQUENCES_FILE      : ${SEQUENCES_FILE:-${EVAL_DIR}/sequences_6eval.inc.sh (default)}"
echo "MOTION_ROOT         : ${MOTION_ROOT}"
echo "MOTION_TAG          : ${MOTION_TAG}"
echo "CO3D_ROOT (GT)      : ${CO3D_ROOT}"
echo "OUT_ROOT            : ${OUT_ROOT}"
echo "DUSt3R ckpt         : ${DUST3R_CKPT}"
echo "Uformer GoPro init  : ${UFORMER_GOPRO}"
echo "Motion finetuned    : ${MOTION_FINETUNED_CKPT}"
echo "================================================================"

require_file() {
  if [[ -L "$1" ]] && [[ ! -e "$1" ]]; then
    echo "ERROR: broken symlink: $1"
    exit 1
  fi
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing file: $1"
    exit 1
  fi
}

require_file "${INF_MOTION}"
require_file "${DUST3R_CKPT}"
require_file "${MOTION_FINETUNED_CKPT}"
require_file "${UFORMER_GOPRO}"

echo "motion_tag,category,seq_id,pipeline,chamfer_distance" > "${RESULT_CSV}.tmp"

run_motion_block() {
  local CATEGORY="$1"
  local SEQ_ID="$2"
  local MDIR="${MOTION_ROOT}/${CATEGORY}/${SEQ_ID}/${MOTION_TAG}"
  local GT_PLY="${CO3D_ROOT}/${CATEGORY}/${SEQ_ID}/pointcloud.ply"
  if [[ ! -d "${MDIR}" ]]; then
    echo "SKIP motion (no dir): ${MDIR}"
    return 0
  fi
  if [[ ! -f "${GT_PLY}" ]]; then
    echo "SKIP motion (no GT): ${GT_PLY}"
    return 0
  fi

  local SAFE_TAG
  SAFE_TAG="${MOTION_TAG//\//_}"
  local TMP_SEQ="${OUT_ROOT}/_tmp_motion_${CATEGORY}_${SEQ_ID}_${SAFE_TAG}"
  rm -rf "${TMP_SEQ}"
  mkdir -p "${TMP_SEQ}"
  ln -sfn "${MDIR}" "${TMP_SEQ}/images"
  ln -sfn "${GT_PLY}" "${TMP_SEQ}/pointcloud.ply"

  for PIPELINE in dust3r uformer_pretrained uformer_finetuned; do
    local ODIR="${OUT_ROOT}/motion/${SAFE_TAG}_${CATEGORY}_${SEQ_ID}_${PIPELINE}"
    mkdir -p "${ODIR}"
    echo ">>> MOTION tag=${MOTION_TAG} ${CATEGORY}/${SEQ_ID} pipeline=${PIPELINE}"
    if [[ "${PIPELINE}" == "dust3r" ]]; then
      "${VENV_PYTHON}" "${INF_MOTION}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline dust3r \
        --quiet
    elif [[ "${PIPELINE}" == "uformer_pretrained" ]]; then
      "${VENV_PYTHON}" "${INF_MOTION}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline uformer_pretrained \
        --uformer_repo "${UFORMER_REPO}" \
        --uformer_pretrained_weights "${UFORMER_GOPRO}" \
        --quiet
    else
      "${VENV_PYTHON}" "${INF_MOTION}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline uformer_finetuned \
        --uformer_repo "${UFORMER_REPO}" \
        --finetuned_weights "${MOTION_FINETUNED_CKPT}" \
        --quiet
    fi
    if [[ -f "${ODIR}/metrics.txt" ]]; then
      local CD
      CD="$(grep -E '^chamfer_distance:' "${ODIR}/metrics.txt" | head -1 | awk '{print $2}')"
      echo "${MOTION_TAG},${CATEGORY},${SEQ_ID},${PIPELINE},${CD}" >> "${RESULT_CSV}.tmp"
    fi
  done
  rm -rf "${TMP_SEQ}"
}

for CATEGORY in "${!SEQUENCES[@]}"; do
  SEQ_ID="${SEQUENCES[$CATEGORY]}"
  run_motion_block "${CATEGORY}" "${SEQ_ID}"
done

mv "${RESULT_CSV}.tmp" "${RESULT_CSV}"

ROW_COUNT="$(($(wc -l < "${RESULT_CSV}") - 1))"
if [[ "${ROW_COUNT}" -lt 1 ]]; then
  echo "ERROR: No Chamfer rows in ${RESULT_CSV} (all skipped or failed)."
  echo "Create motion trees under: ${MOTION_ROOT}/<category>/<seq>/${MOTION_TAG}/"
  echo "or fix CO3D_ROOT / SEQUENCES_FILE / MOTION_ROOT."
  exit 1
fi

"${VENV_PYTHON}" - <<PY
import csv
from collections import defaultdict
from pathlib import Path

csv_path = Path("${RESULT_CSV}")
rows = list(csv.DictReader(csv_path.open()))
out_path = csv_path.with_suffix(".summary.txt")

acc = defaultdict(list)
for r in rows:
    try:
        v = float(r["chamfer_distance"])
    except (TypeError, ValueError):
        continue
    acc[r["pipeline"]].append(v)

lines = [str(csv_path), "=" * 72, "Mean Chamfer over sequences (lower is better)", f"motion_tag=${MOTION_TAG}", ""]
pls = ["dust3r", "uformer_pretrained", "uformer_finetuned"]
for p in pls:
    vals = acc.get(p, [])
    if vals:
        lines.append(f"  {p} = {sum(vals)/len(vals):.8f}  (n={len(vals)})")
    else:
        lines.append(f"  {p} = NA")
lines.append("")
lines.append("Full rows: see CSV.")
out_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Wrote: {out_path}")
PY

echo "Done: $(date)"
echo "Results: ${RESULT_CSV}"
