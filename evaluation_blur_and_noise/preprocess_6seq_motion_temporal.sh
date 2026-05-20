#!/usr/bin/env bash
# Build GoPro-like temporal-averaged motion blur for the 6 eval sequences
# (same as sequences_6eval.inc.sh). Output:
#   MOTION_ROOT/<cat>/<seq_id>/MOTION_TAG/frame*.jpg
#
# Prerequisite: CO3D_ROOT has images/ (and matching preprocess) for each sequence.
#
# Usage (YSU):
#   CO3D_ROOT=/mnt/weka/hminasyan/data/co3d_processed \
#   MOTION_ROOT=/mnt/weka/hminasyan/outputs/degraded_frames_motion_6seq \
#   MOTION_TAG=temporal_avg_w11_gopro_like \
#     bash evaluation_blur_and_noise/preprocess_6seq_motion_temporal.sh
#
# Or Slurm: sbatch -p all --mem=32G --time=24:00:00 .../preprocess_6seq_motion_temporal.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == /var/spool/slurmd/* ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
else
  : "${PROJECT_DIR:=$(cd "${SCRIPT_DIR}/.." && pwd)}"
fi
cd "${PROJECT_DIR}"

if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
elif [[ -f /home/asds/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/asds/miniforge3/etc/profile.d/conda.sh
  conda activate co3d_env
fi

: "${CO3D_ROOT:=${PROJECT_DIR}/data/co3d_processed}"
: "${MOTION_ROOT:=${PROJECT_DIR}/outputs/degraded_frames_motion_6seq}"
: "${MOTION_TAG:=temporal_avg_w11_gopro_like}"
: "${WINDOW:=11}"

PY="${SCRIPT_DIR}/generate_temporal_avg_motion_6seq.py"
[[ -f "${PY}" ]] || { echo "ERROR: missing ${PY}"; exit 1; }

echo "CO3D_ROOT=$CO3D_ROOT"
echo "MOTION_ROOT=$MOTION_ROOT  MOTION_TAG=$MOTION_TAG  WINDOW=$WINDOW"
python3 "${PY}" \
  --co3d_root "${CO3D_ROOT}" \
  --motion_root "${MOTION_ROOT}" \
  --motion_tag "${MOTION_TAG}" \
  --window "${WINDOW}"
echo "Next: run_eval_6seq_motion_chamfer.sh with same MOTION_ROOT, MOTION_TAG, CO3D_ROOT"
