#!/usr/bin/env bash
# Experiment (ASDS / GPU): 6 CO3D sequences — bottle, cup, hydrant, teddybear, toybus, toytrain
#
# Blur σ ∈ {5,10,20,30} — pipelines: DUSt3R | DeblurDiNAT (pretrained) + DUSt3R | finetuned DeblurDiNAT + DUSt3R
# Noise σ ∈ {30,50,70} (same as joint Uformer+DUSt3R finetuning) — DUSt3R | Uformer (pretrained) + DUSt3R | finetuned Uformer + DUSt3R
#
# DUSt3R ViT-L **224** + IMAGE_SIZE=224. Metrics: Chamfer vs GT point cloud; summary = mean over the 6 sequences
# per (kind, σ, pipeline).
#
# Data prep (once):  bash evaluation_blur_and_noise/preprocess_6seq_blur_noise_eval_asds.sh
# Then:              sbatch evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer.sh
#
# Overrides: DUST3R_CKPT, IMAGE_SIZE, CO3D_ROOT, BLUR_ROOT, NOISE_ROOT, *_CKPT paths,
#             NOISE_EVAL_SIGMAS (default: 30 50 70, match finetuning).
#
# Usage (from anywhere):
#   bash /path/to/project_Hayk_Minasyan/evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer.sh
# Or:
#   cd /path/to/project_Hayk_Minasyan/evaluation_blur_and_noise && ./run_eval_6seq_blur_noise_chamfer.sh
#
#SBATCH --job-name=eval6seq_bn
# Do not hardcode --chdir to one cluster (YSU has no /home/asds/...). Submit from the repo root:
#   cd /path/to/project_Hayk_Minasyan && sbatch evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer.sh
# Or: sbatch --chdir=/path/to/project_Hayk_Minasyan .../run_eval_6seq_blur_noise_chamfer.sh
# Partition: override at submit time if needed (ASDS: -p a100 ; YSU: -p research).
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
# Written to submit working directory (project root). Avoid logs/ subdir here so Slurm can create the file even if logs/ did not exist yet.
#SBATCH --output=eval_6seq_blur_noise_%j.log
#SBATCH --error=eval_6seq_blur_noise_%j.err

set -euo pipefail

# When submitted via sbatch, Slurm runs a *copy* of this script under /var/spool/slurmd/...,
# so BASH_SOURCE must not be used to infer PROJECT_DIR (would try to mkdir under /var/spool).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == /var/spool/slurmd/* ]]; then
  # Prefer explicit PROJECT_DIR from `sbatch --export=ALL,PROJECT_DIR=...` (e.g. YSU).
  if [[ -n "${PROJECT_DIR:-}" && -d "${PROJECT_DIR}/dust3r" ]]; then
    :
  elif [[ -d "$(pwd)/dust3r" ]]; then
    PROJECT_DIR="$(pwd -P)"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/dust3r" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    echo "ERROR: Under sbatch, could not find project root (need ./dust3r). Use:"
    echo "  cd /path/to/project_Hayk_Minasyan && sbatch evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer.sh"
    echo "Or: sbatch --export=ALL,PROJECT_DIR=/path/to/project_Hayk_Minasyan ..."
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

# Inference entry points (symlinks in this folder → finetune_*)
INF_BLUR="${EVAL_DIR}/deblurdinat_run_inference.py"
INF_NOISE="${EVAL_DIR}/uformer_run_inference.py"

# Override on YSU weka: SEQUENCES_FILE=.../sequences_6eval_weka.inc.sh
# shellcheck source=sequences_6eval.inc.sh
# shellcheck disable=SC1091
source "${SEQUENCES_FILE:-${EVAL_DIR}/sequences_6eval.inc.sh}"

: "${N_FRAMES:=20}"
# Match DUSt3R ViT-L 224 training & finetuned checkpoints in this project.
: "${IMAGE_SIZE:=224}"

: "${DUST3R_DIR:=${PROJECT_DIR}/dust3r}"
: "${DUST3R_CKPT:=${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"

: "${CO3D_ROOT:=${PROJECT_DIR}/data/co3d_processed_10cat}"
# On asds, blur + noise for the 6 eval sequences often live under one tree (override if needed):
: "${BLUR_ROOT:=${PROJECT_DIR}/outputs/degraded_frames}"
: "${NOISE_ROOT:=${PROJECT_DIR}/outputs/degraded_frames}"

: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
: "${DEBLURDINAT_PRETRAINED:=${DEBLURDINAT_REPO}/results/DeblurDiNATL/models/DeblurDiNATL.pth}"
: "${BLUR_FINETUNED_CKPT:=${PROJECT_DIR}/finetune_blur_runs/deblurdinat_dust3r_asds_224_joint_5_10_20_30_from_dust3r/joint_sigmas_5_10_20_30/checkpoint-best-val.pth}"
# ASDS default path is often absent on YSU; try typical weka joint run (5–50 sigmas).
if [[ ! -f "${BLUR_FINETUNED_CKPT}" ]]; then
  _BLUR_FT_FALLBACK="/mnt/weka/hminasyan/finetune_blur_runs/deblurdinat_dust3r_ysu_224_joint_5_10_20_30_50_50ep/joint_sigmas_5_10_20_30_50/checkpoint-best-val.pth"
  if [[ -f "${_BLUR_FT_FALLBACK}" ]]; then
    BLUR_FINETUNED_CKPT="${_BLUR_FT_FALLBACK}"
  fi
fi
unset _BLUR_FT_FALLBACK 2>/dev/null || true

: "${UFORMER_REPO:=${PROJECT_DIR}/Uformer}"
: "${UFORMER_PRETRAINED:=${UFORMER_REPO}/logs/denoising/SIDD/Uformer_B/models/model_best.pth}"
: "${NOISE_FINETUNED_CKPT:=${PROJECT_DIR}/finetune_noise_runs/uformer_dust3r_asds_224_recon_l05_lr2e4_wu3_wd02_random50/joint_sigmas_30_50_70/checkpoint-best-val.pth}"

: "${OUT_ROOT:=${PROJECT_DIR}/outputs/dust3r_eval_6seq_blur_noise_$(date +%Y%m%d)}"
: "${RESULT_CSV:=${OUT_ROOT}/chamfer_results.csv}"

# croco first so `import models.dpt_block` resolves to dust3r/croco/models (not DeblurDiNAT/models).
# Do not put DeblurDiNAT on PYTHONPATH here — it shadows croco's `models`; finetune_blur/deblurdinat/model.py adds it when needed.
export PYTHONPATH="${PROJECT_DIR}/dust3r/croco:${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${UFORMER_REPO}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}"

echo "================================================================"
echo "6-seq blur + noise Chamfer eval | $(date) | $(hostname)"
echo "DUSt3R 224 model    : ${DUST3R_CKPT}"
echo "IMAGE_SIZE (px)     : ${IMAGE_SIZE}  (must match 224 finetuning)"
echo "EVAL_DIR            : ${EVAL_DIR}"
echo "PROJECT_DIR         : ${PROJECT_DIR}"
echo "CO3D_ROOT           : ${CO3D_ROOT}"
echo "BLUR_ROOT           : ${BLUR_ROOT}"
echo "NOISE_ROOT          : ${NOISE_ROOT}"
echo "OUT_ROOT            : ${OUT_ROOT}"
echo "RESULT_CSV          : ${RESULT_CSV}"
echo "Deblur pretrained   : ${DEBLURDINAT_PRETRAINED}"
echo "Deblur finetuned    : ${BLUR_FINETUNED_CKPT}"
echo "Uformer pretrained  : ${UFORMER_PRETRAINED}"
echo "Noise finetuned     : ${NOISE_FINETUNED_CKPT}"
echo "================================================================"

require_file() {
  if [[ -L "$1" ]] && [[ ! -e "$1" ]]; then
    echo "ERROR: broken symlink (target missing): $1 -> $(readlink "$1" 2>/dev/null || echo '?')"
    echo "Fix: rsync the target tree from ASDS, or from repo root:"
    echo "  ln -sf ../finetune_noise/run_inference.py evaluation_blur_and_noise/uformer_run_inference.py"
    echo "  ln -sf ../finetune_blur/deblurdinat/run_inference.py evaluation_blur_and_noise/deblurdinat_run_inference.py"
    exit 1
  fi
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing file: $1"
    exit 1
  fi
}

require_file "${INF_BLUR}"
require_file "${INF_NOISE}"
require_file "${DUST3R_CKPT}"
require_file "${DEBLURDINAT_PRETRAINED}"
require_file "${BLUR_FINETUNED_CKPT}"
require_file "${UFORMER_PRETRAINED}"
require_file "${NOISE_FINETUNED_CKPT}"

# CSV header
echo "kind,sigma,category,seq_id,pipeline,chamfer_distance" > "${RESULT_CSV}.tmp"

run_blur_block() {
  local S="$1"
  local CATEGORY="$2"
  local SEQ_ID="$3"
  local BLUR_DIR="${BLUR_ROOT}/${CATEGORY}/${SEQ_ID}/blur_s${S}"
  local GT_PLY="${CO3D_ROOT}/${CATEGORY}/${SEQ_ID}/pointcloud.ply"
  if [[ ! -d "${BLUR_DIR}" ]]; then
    echo "SKIP blur (no dir): ${BLUR_DIR}"
    return 0
  fi
  if [[ ! -f "${GT_PLY}" ]]; then
    echo "SKIP blur (no GT): ${GT_PLY}"
    return 0
  fi

  local TMP_SEQ="${OUT_ROOT}/_tmp_blur_${CATEGORY}_${SEQ_ID}_s${S}"
  rm -rf "${TMP_SEQ}"
  mkdir -p "${TMP_SEQ}"
  ln -sfn "${BLUR_DIR}" "${TMP_SEQ}/images"
  ln -sfn "${GT_PLY}" "${TMP_SEQ}/pointcloud.ply"

  for PIPELINE in dust3r deblur_pretrained deblur_finetuned; do
    local TAG="blur_s${S}_${CATEGORY}_${SEQ_ID}_${PIPELINE}"
    local ODIR="${OUT_ROOT}/blur/${TAG}"
    mkdir -p "${ODIR}"
    echo ">>> BLUR sigma=${S} ${CATEGORY}/${SEQ_ID} pipeline=${PIPELINE}"
    if [[ "${PIPELINE}" == "dust3r" ]]; then
      "${VENV_PYTHON}" "${INF_BLUR}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline dust3r \
        --quiet
    elif [[ "${PIPELINE}" == "deblur_pretrained" ]]; then
      "${VENV_PYTHON}" "${INF_BLUR}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline deblur_pretrained \
        --deblurdinat_repo "${DEBLURDINAT_REPO}" \
        --deblurdinat_pretrained_ckpt "${DEBLURDINAT_PRETRAINED}" \
        --quiet
    else
      "${VENV_PYTHON}" "${INF_BLUR}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline deblur_finetuned \
        --deblurdinat_repo "${DEBLURDINAT_REPO}" \
        --finetuned_weights "${BLUR_FINETUNED_CKPT}" \
        --quiet
    fi
    if [[ -f "${ODIR}/metrics.txt" ]]; then
      local CD
      CD="$(grep -E '^chamfer_distance:' "${ODIR}/metrics.txt" | head -1 | awk '{print $2}')"
      echo "blur,${S},${CATEGORY},${SEQ_ID},${PIPELINE},${CD}" >> "${RESULT_CSV}.tmp"
    fi
  done
  rm -rf "${TMP_SEQ}"
}

run_noise_block() {
  local S="$1"
  local CATEGORY="$2"
  local SEQ_ID="$3"
  local NOISE_DIR="${NOISE_ROOT}/${CATEGORY}/${SEQ_ID}/noise_s${S}"
  local GT_PLY="${CO3D_ROOT}/${CATEGORY}/${SEQ_ID}/pointcloud.ply"
  if [[ ! -d "${NOISE_DIR}" ]]; then
    echo "SKIP noise (no dir): ${NOISE_DIR}"
    return 0
  fi
  if [[ ! -f "${GT_PLY}" ]]; then
    echo "SKIP noise (no GT): ${GT_PLY}"
    return 0
  fi

  local TMP_SEQ="${OUT_ROOT}/_tmp_noise_${CATEGORY}_${SEQ_ID}_s${S}"
  rm -rf "${TMP_SEQ}"
  mkdir -p "${TMP_SEQ}"
  ln -sfn "${NOISE_DIR}" "${TMP_SEQ}/images"
  ln -sfn "${GT_PLY}" "${TMP_SEQ}/pointcloud.ply"

  for PIPELINE in dust3r uformer_pretrained uformer_finetuned; do
    local TAG="noise_s${S}_${CATEGORY}_${SEQ_ID}_${PIPELINE}"
    local ODIR="${OUT_ROOT}/noise/${TAG}"
    mkdir -p "${ODIR}"
    echo ">>> NOISE sigma=${S} ${CATEGORY}/${SEQ_ID} pipeline=${PIPELINE}"
    if [[ "${PIPELINE}" == "dust3r" ]]; then
      "${VENV_PYTHON}" "${INF_NOISE}" \
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
      "${VENV_PYTHON}" "${INF_NOISE}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline uformer_pretrained \
        --uformer_repo "${UFORMER_REPO}" \
        --uformer_pretrained_weights "${UFORMER_PRETRAINED}" \
        --quiet
    else
      "${VENV_PYTHON}" "${INF_NOISE}" \
        --sequence_dir "${TMP_SEQ}" \
        --dust3r_dir "${DUST3R_DIR}" \
        --dust3r_ckpt "${DUST3R_CKPT}" \
        --output_dir "${ODIR}" \
        --image_size "${IMAGE_SIZE}" \
        --n_frames "${N_FRAMES}" \
        --device cuda \
        --pipeline uformer_finetuned \
        --uformer_repo "${UFORMER_REPO}" \
        --finetuned_weights "${NOISE_FINETUNED_CKPT}" \
        --quiet
    fi
    if [[ -f "${ODIR}/metrics.txt" ]]; then
      local CD
      CD="$(grep -E '^chamfer_distance:' "${ODIR}/metrics.txt" | head -1 | awk '{print $2}')"
      echo "noise,${S},${CATEGORY},${SEQ_ID},${PIPELINE},${CD}" >> "${RESULT_CSV}.tmp"
    fi
  done
  rm -rf "${TMP_SEQ}"
}

 : "${BLUR_EVAL_SIGMAS:=5 10 20 30}"
# shellcheck disable=SC2206
BLUR_EVAL_ARR=(${BLUR_EVAL_SIGMAS})
for S in "${BLUR_EVAL_ARR[@]}"; do
  for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    run_blur_block "${S}" "${CATEGORY}" "${SEQ_ID}"
  done
done

# Noise levels match finetune_noise (joint_sigmas_30_50_70). Override: NOISE_EVAL_SIGMAS="30 50 70"
: "${NOISE_EVAL_SIGMAS:=30 50 70}"
# shellcheck disable=SC2206
NOISE_EVAL_ARR=(${NOISE_EVAL_SIGMAS})
for S in "${NOISE_EVAL_ARR[@]}"; do
  for CATEGORY in "${!SEQUENCES[@]}"; do
    SEQ_ID="${SEQUENCES[$CATEGORY]}"
    run_noise_block "${S}" "${CATEGORY}" "${SEQ_ID}"
  done
done

mv "${RESULT_CSV}.tmp" "${RESULT_CSV}"

ROW_COUNT="$(($(wc -l < "${RESULT_CSV}") - 1))"
if [[ "${ROW_COUNT}" -lt 1 ]]; then
  echo "ERROR: No Chamfer rows in ${RESULT_CSV} (all pipelines skipped or failed)."
  echo "Run: bash evaluation_blur_and_noise/preprocess_6seq_blur_noise_eval_asds.sh"
  exit 1
fi

"${VENV_PYTHON}" - <<PY
import csv
from collections import defaultdict
from pathlib import Path

csv_path = Path("${RESULT_CSV}")
rows = list(csv.DictReader(csv_path.open()))
out_md = csv_path.with_suffix(".summary.txt")

acc = defaultdict(list)
for r in rows:
    try:
        v = float(r["chamfer_distance"])
    except (TypeError, ValueError):
        continue
    acc[(r["kind"], r["sigma"], r["pipeline"])].append(v)

lines = [str(csv_path), "=" * 72, "Mean Chamfer over sequences (lower is better)", ""]

blur_pl = ["dust3r", "deblur_pretrained", "deblur_finetuned"]
noise_pl = ["dust3r", "uformer_pretrained", "uformer_finetuned"]

blur_sigmas = [s for s in "${BLUR_EVAL_SIGMAS}".split() if s.strip()]
if not blur_sigmas:
    blur_sigmas = ["5", "10", "20", "30"]
noise_sigmas = [s for s in "${NOISE_EVAL_SIGMAS}".split() if s.strip()]
if not noise_sigmas:
    noise_sigmas = ["30", "50", "70"]

for kind, sigmas, pls in (
    ("blur", blur_sigmas, blur_pl),
    ("noise", noise_sigmas, noise_pl),
):
    lines.append(f"[{kind.upper()}]")
    for s in sigmas:
        parts = [f"sigma={s}"]
        for p in pls:
            vals = acc.get((kind, s, p), [])
            if vals:
                parts.append(f"{p}={sum(vals)/len(vals):.8f}(n={len(vals)})")
            else:
                parts.append(f"{p}=NA")
        lines.append("  " + " | ".join(parts))
    lines.append("")

lines.append("Full per-sequence rows: see CSV.")
out_md.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Wrote summary: {out_md}")
PY

echo ""
echo "Done: $(date)"
echo "Results: ${RESULT_CSV}"
