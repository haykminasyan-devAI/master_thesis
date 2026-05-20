#!/usr/bin/env bash
# Generate blurred and noisy frame trees for the *same six sequences* as the Chamfer eval.
# Layout must match run_eval_6seq_blur_noise_chamfer.sh:
#   BLUR_ROOT/<category>/<seq_id>/blur_s{5,10,20,30}/
#   NOISE_ROOT/<category>/<seq_id>/noise_s{30,50,70}/  (match noise finetuning sigmas)
# Inputs: preprocessed DUSt3R CO3D crops — CO3D_ROOT/<category>/<seq_id>/images/
#
# Prerequisite: CO3D_ROOT must already contain those six categories with pointcloud + images
# (from dust3r datasets_preprocess / your co3d_processed tree).
#
# Usage (ASDS login or Slurm):
#   cd /home/asds/project_Hayk_Minasyan
#   bash evaluation_blur_and_noise/preprocess_6seq_blur_noise_eval_asds.sh
#   # or: sbatch evaluation_blur_and_noise/preprocess_6seq_blur_noise_eval_asds.sh
#
# Then run inference + Chamfer:
#   sbatch evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer.sh

#SBATCH --job-name=prep6seq_bn
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/prep_6seq_blur_noise_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/prep_6seq_blur_noise_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == /var/spool/slurmd/* ]]; then
  if [[ -d "$(pwd)/dust3r" ]]; then
    PROJECT_DIR="$(pwd -P)"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/dust3r" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_DIR="/home/asds/project_Hayk_Minasyan"
  fi
  EVAL_DIR="${PROJECT_DIR}/evaluation_blur_and_noise"
else
  PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
  EVAL_DIR="${SCRIPT_DIR}"
fi
cd "${PROJECT_DIR}"

# Override on YSU weka: SEQUENCES_FILE=.../sequences_6eval_weka.inc.sh
# shellcheck source=/dev/null
# shellcheck disable=SC1091
source "${SEQUENCES_FILE:-${EVAL_DIR}/sequences_6eval.inc.sh}"

if [[ -f /home/asds/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/asds/miniforge3/etc/profile.d/conda.sh
  conda activate co3d_env
fi
PYTHON="${PYTHON:-python3}"

CO3D_ROOT="${CO3D_ROOT:-${PROJECT_DIR}/data/co3d_processed_10cat}"
BLUR_ROOT="${BLUR_ROOT:-${PROJECT_DIR}/outputs/degraded_frames_10cat}"
NOISE_ROOT="${NOISE_ROOT:-${PROJECT_DIR}/outputs/noisy_frames_10cat}"
BLUR_SIGMAS=(5 10 20 30)
# Match finetune_noise joint training (30 50 70)
NOISE_SIGMAS=(30 50 70)

DEGRADE_BLUR="${PROJECT_DIR}/finetune_blur/degrade_blur.py"
# Prefer copy next to this script (YSU / minimal checkouts may lack scripts/gaussian_noise_and_blur_exps/)
DEGRADE_NOISE="${EVAL_DIR}/degrade.py"
if [[ ! -f "${DEGRADE_NOISE}" ]]; then
  DEGRADE_NOISE="${PROJECT_DIR}/scripts/gaussian_noise_and_blur_exps/degrade.py"
fi

echo "================================================================"
echo "6-seq blur + noise frame prep | $(date) | $(hostname)"
echo "CO3D_ROOT (clean crops) : ${CO3D_ROOT}"
echo "BLUR_ROOT (output)      : ${BLUR_ROOT}"
echo "NOISE_ROOT (output)     : ${NOISE_ROOT}"
echo "Blur σ                  : ${BLUR_SIGMAS[*]}"
echo "Noise σ (pixel scale)   : ${NOISE_SIGMAS[*]}"
echo "================================================================"

[[ -f "${DEGRADE_BLUR}" ]] || { echo "ERROR: missing ${DEGRADE_BLUR}"; exit 1; }
[[ -f "${DEGRADE_NOISE}" ]] || { echo "ERROR: missing ${DEGRADE_NOISE}"; exit 1; }

mkdir -p "${BLUR_ROOT}" "${NOISE_ROOT}"

MISSING=0
for CATEGORY in "${!SEQUENCES[@]}"; do
  SEQ_ID="${SEQUENCES[$CATEGORY]}"
  IMGS="${CO3D_ROOT}/${CATEGORY}/${SEQ_ID}/images"
  PLY="${CO3D_ROOT}/${CATEGORY}/${SEQ_ID}/pointcloud.ply"
  if [[ ! -d "${IMGS}" ]]; then
    echo "ERROR: missing processed images (run CO3D preprocess for this seq first): ${IMGS}"
    MISSING=1
  fi
  if [[ ! -f "${PLY}" ]]; then
    echo "ERROR: missing GT point cloud: ${PLY}"
    MISSING=1
  fi
done
if [[ "${MISSING}" -ne 0 ]]; then
  echo "Fix CO3D_ROOT tree, then re-run this script."
  exit 1
fi

for CATEGORY in "${!SEQUENCES[@]}"; do
  SEQ_ID="${SEQUENCES[$CATEGORY]}"
  PROC_IMGS="${CO3D_ROOT}/${CATEGORY}/${SEQ_ID}/images"
  for SIGMA in "${BLUR_SIGMAS[@]}"; do
    OUT="${BLUR_ROOT}/${CATEGORY}/${SEQ_ID}/blur_s${SIGMA}"
    if [[ -d "${OUT}" ]] && [[ "$(find "${OUT}" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]]; then
      echo "  [skip] blur ${CATEGORY}/${SEQ_ID}/blur_s${SIGMA}"
    else
      echo "  [gen]  blur ${CATEGORY}/${SEQ_ID}/blur_s${SIGMA}"
      mkdir -p "${OUT}"
      "${PYTHON}" "${DEGRADE_BLUR}" \
        --images_dir "${PROC_IMGS}" \
        --output_dir "${OUT}" \
        --blur_sigma "${SIGMA}"
    fi
  done
  for STD in "${NOISE_SIGMAS[@]}"; do
    OUT="${NOISE_ROOT}/${CATEGORY}/${SEQ_ID}/noise_s${STD}"
    if [[ -d "${OUT}" ]] && [[ "$(find "${OUT}" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]]; then
      echo "  [skip] noise ${CATEGORY}/${SEQ_ID}/noise_s${STD}"
    else
      echo "  [gen]  noise ${CATEGORY}/${SEQ_ID}/noise_s${STD}"
      mkdir -p "${OUT}"
      "${PYTHON}" "${DEGRADE_NOISE}" \
        --images_dir "${PROC_IMGS}" \
        --output_dir "${OUT}" \
        --mode noise \
        --noise_std "${STD}" \
        --seed 42
    fi
  done
done

echo ""
echo "Done: $(date)"
echo "Next: sbatch evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer.sh"
