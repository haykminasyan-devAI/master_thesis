#!/usr/bin/env bash
#SBATCH --job-name=co3d_noise10cat
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/co3d_noise10cat_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/co3d_noise10cat_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
PYTHON="${PYTHON:-python3}"

CO3D_PROCESSED="${CO3D_PROCESSED:-${PROJECT_DIR}/data/co3d_processed_10cat}"
NOISE_ROOT="${NOISE_ROOT:-${PROJECT_DIR}/outputs/noisy_frames_10cat}"
NOISE_SIGMAS=(${NOISE_SIGMAS:-30 50 70})

CATEGORIES=(apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot)
SEQ_JSON="${PROJECT_DIR}/finetune_blur/sequences_10cat.json"

echo "============================================================"
echo "10-cat noise preprocessing | $(date) | $(hostname)"
echo "CO3D processed : ${CO3D_PROCESSED}"
echo "NOISE root     : ${NOISE_ROOT}"
echo "Noise sigmas   : ${NOISE_SIGMAS[*]}"
echo "============================================================"

if [[ ! -f "${SEQ_JSON}" ]]; then
  echo "ERROR: sequence manifest not found: ${SEQ_JSON}"
  exit 1
fi

mkdir -p "${NOISE_ROOT}"

declare -A SEQ_ID
for CAT in "${CATEGORIES[@]}"; do
  SID="$("${PYTHON}" -c "import json; d=json.load(open('${SEQ_JSON}')); print(d.get('${CAT}',''))")"
  if [[ -z "${SID}" ]]; then
    echo "ERROR: missing sequence for ${CAT} in ${SEQ_JSON}"
    exit 1
  fi
  SEQ_ID["${CAT}"]="${SID}"
done

for CAT in "${CATEGORIES[@]}"; do
  SID="${SEQ_ID[$CAT]}"
  SRC_IMGS="${CO3D_PROCESSED}/${CAT}/${SID}/images"
  if [[ ! -d "${SRC_IMGS}" ]]; then
    echo "ERROR: missing processed images: ${SRC_IMGS}"
    exit 1
  fi
  for S in "${NOISE_SIGMAS[@]}"; do
    OUT_DIR="${NOISE_ROOT}/${CAT}/${SID}/noise_s${S}"
    if [[ -d "${OUT_DIR}" ]] && [[ "$(ls -A "${OUT_DIR}" 2>/dev/null | wc -l)" -gt 0 ]]; then
      echo "  [skip] ${CAT}/${SID}/noise_s${S}"
    else
      echo "  [gen ] ${CAT}/${SID}/noise_s${S}"
      mkdir -p "${OUT_DIR}"
      "${PYTHON}" scripts/gaussian_noise_and_blur_exps/degrade.py \
        --images_dir "${SRC_IMGS}" \
        --output_dir "${OUT_DIR}" \
        --mode noise \
        --noise_std "${S}" \
        --seed 42
    fi
  done
done

echo ""
echo "Done: $(date)"
echo "Noise root ready: ${NOISE_ROOT}"
