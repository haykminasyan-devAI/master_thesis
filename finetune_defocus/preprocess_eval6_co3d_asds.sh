#!/usr/bin/env bash
#SBATCH --job-name=co3d_eval6_prep
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/co3d_eval6_prep_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/co3d_eval6_prep_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
CO3D_RAW="${CO3D_RAW:-${PROJECT_DIR}/data/co3d}"
CO3D_PROCESSED="${CO3D_PROCESSED:-${PROJECT_DIR}/data/co3d_processed_10cat}"
EVAL_CATEGORIES="${EVAL_CATEGORIES:-bottle cup donut teddybear couch toytrain}"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
PYTHON="$(command -v python3)"

mkdir -p "${PROJECT_DIR}/logs" "${CO3D_PROCESSED}"
export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}/dust3r/datasets_preprocess"
for cat in ${EVAL_CATEGORIES}; do
  if [ -d "${CO3D_PROCESSED}/${cat}" ]; then
    echo "[skip] ${cat} exists in processed root"
    continue
  fi
  echo "[proc] ${cat}"
  "${PYTHON}" preprocess_co3d.py \
    --co3d_dir "${CO3D_RAW}" \
    --output_dir "${CO3D_PROCESSED}" \
    --category "${cat}" \
    --num_sequences_per_object 50 \
    --seed 42
done

echo "Done preprocessing eval categories."
