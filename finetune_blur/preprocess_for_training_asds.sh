#!/usr/bin/env bash
#SBATCH --job-name=co3d_preprocess
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/co3d_preprocess_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/co3d_preprocess_%j.err
#
# One-time CO3D preprocessing for ASDS cluster.
# Run BEFORE submitting train_dust3r_deblur_asds.sh.
#
# What it does:
#   1. Discovers one sequence per category (sequences_10cat.json)
#   2. Runs preprocess_co3d.py -> co3d_processed_10cat/ (images, depths, masks .npz)
#   3. Generates blurred frames (sigma=5,10,20) under degraded_frames_10cat/
#   4. Writes selected_seqs_train.json + selected_seqs_test.json
#
# Usage (submit as SLURM job):
#   cd /home/asds/project_Hayk_Minasyan
#   sbatch finetune_blur/preprocess_for_training_asds.sh
#
# Or run interactively:
#   bash finetune_blur/preprocess_for_training_asds.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env

CO3D_DIR="${CO3D_DIR:-${PROJECT_DIR}/data/co3d}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data/co3d_processed_10cat}"
DEGRADED_ROOT="${DEGRADED_ROOT:-${PROJECT_DIR}/outputs/degraded_frames_10cat}"
PYTHON="${PYTHON:-python3}"
DUST3R_DIR="${PROJECT_DIR}/dust3r"
SCRIPT_DIR="${PROJECT_DIR}/finetune_blur"

CATEGORIES=(apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot)
BLUR_SIGMAS=(5 10 20)

SEQ_JSON="${SCRIPT_DIR}/sequences_10cat.json"

export PYTHONPATH="${DUST3R_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"

echo "================================================================"
echo "ASDS — CO3D Preprocessing  |  $(date)  |  $(hostname)"
echo "CO3D raw     : ${CO3D_DIR}"
echo "CO3D output  : ${OUTPUT_DIR}"
echo "Degraded root: ${DEGRADED_ROOT}"
echo "Sigmas       : ${BLUR_SIGMAS[*]}"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}" "${DEGRADED_ROOT}" "${PROJECT_DIR}/logs"

# ── 1. Discover one sequence per category ─────────────────────────────────────
if [ -f "${SEQ_JSON}" ]; then
  echo ""
  echo ">>> Using existing sequence manifest: ${SEQ_JSON}"
else
  echo ""
  echo ">>> Discovering one sequence per category -> ${SEQ_JSON}"
  "${PYTHON}" "${SCRIPT_DIR}/discover_sequences_10cat.py" \
    --co3d_dir "${CO3D_DIR}" \
    --out "${SEQ_JSON}"
fi

declare -A SEQ_ID
for CAT in "${CATEGORIES[@]}"; do
  SID="$("${PYTHON}" -c "import json; d=json.load(open('${SEQ_JSON}')); print(d.get('${CAT}',''))")"
  if [[ -z "${SID}" ]]; then
    echo "ERROR: missing sequence for ${CAT} in ${SEQ_JSON}"
    exit 1
  fi
  SEQ_ID["${CAT}"]="${SID}"
  echo "  ${CAT} -> ${SID}"
done

# ── 2. Write set_lists/fewview_train_custom.json per category ─────────────────
echo ""
echo ">>> Writing set_lists/fewview_train_custom.json per category ..."
for CAT in "${CATEGORIES[@]}"; do
  SET_LIST_DIR="${CO3D_DIR}/${CAT}/set_lists"
  SID="${SEQ_ID[$CAT]}"
  mkdir -p "${SET_LIST_DIR}"

  "${PYTHON}" - <<PY
import gzip, json, os

cat = "${CAT}"
co3d_dir = "${CO3D_DIR}"
keep = {"${SID}"}

frame_file = os.path.join(co3d_dir, cat, "frame_annotations.jgz")
with gzip.open(frame_file, "r") as fin:
    frame_data = json.loads(fin.read())

entries = []
for f in frame_data:
    seq_name = f.get("sequence_name")
    if seq_name not in keep:
        continue
    frame_number = f.get("frame_number")
    image_path = f.get("image", {}).get("path")
    if image_path is None:
        continue
    entries.append([seq_name, frame_number, image_path])

entries.sort(key=lambda x: (x[0], x[1]))
out = os.path.join(co3d_dir, cat, "set_lists", "fewview_train_custom.json")
with open(out, "w") as f:
    json.dump({"train": entries, "test": entries}, f)
print(f"  [gen] {cat}/{seq_name}: {len(entries)} entries -> {out}")
PY
done

# ── 3. Run preprocess_co3d.py (DUSt3R preprocessor) ──────────────────────────
echo ""
echo ">>> Running preprocess_co3d.py for each category ..."
cd "${DUST3R_DIR}/datasets_preprocess"

for CAT in "${CATEGORIES[@]}"; do
  PER_CAT_FLAG="${OUTPUT_DIR}/${CAT}/selected_seqs_train.json"
  if [ -f "${PER_CAT_FLAG}" ]; then
    echo "  [skip] ${CAT} (already preprocessed)"
  else
    echo "  [proc] ${CAT} ..."
    rm -f "${OUTPUT_DIR}/selected_seqs_train.json" "${OUTPUT_DIR}/selected_seqs_test.json"
    "${PYTHON}" preprocess_co3d.py \
      --co3d_dir "${CO3D_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --category "${CAT}" \
      --num_sequences_per_object 50 \
      --seed 42
  fi
done

cd "${PROJECT_DIR}"

# ── 4. Generate blurred frames ────────────────────────────────────────────────
echo ""
echo ">>> Generating blurred frames from preprocessed images ..."
for CAT in "${CATEGORIES[@]}"; do
  SID="${SEQ_ID[$CAT]}"
  PROC_IMGS="${OUTPUT_DIR}/${CAT}/${SID}/images"
  if [ ! -d "${PROC_IMGS}" ]; then
    echo "  [WARN] ${CAT}/${SID}/images not found (preprocessing may have filtered this sequence) — skipping blur"
    continue
  fi
  for SIGMA in "${BLUR_SIGMAS[@]}"; do
    OUT="${DEGRADED_ROOT}/${CAT}/${SID}/blur_s${SIGMA}"
    if [ -d "${OUT}" ] && [ "$(ls -A "${OUT}" 2>/dev/null | wc -l)" -gt 0 ]; then
      echo "  [skip] ${CAT}/${SID}/blur_s${SIGMA}"
    else
      echo "  [gen]  ${CAT}/${SID}/blur_s${SIGMA} ..."
      mkdir -p "${OUT}"
      "${PYTHON}" "${SCRIPT_DIR}/degrade_blur.py" \
        --images_dir "${PROC_IMGS}" \
        --output_dir "${OUT}" \
        --blur_sigma "${SIGMA}"
    fi
  done
done

# ── 5. Write selected_seqs_*.json ─────────────────────────────────────────────
echo ""
echo ">>> Writing selected_seqs_train.json / selected_seqs_test.json ..."
"${PYTHON}" "${SCRIPT_DIR}/make_selected_seqs_10cat.py" \
  --co3d_processed "${OUTPUT_DIR}" \
  --sequences_json "${SEQ_JSON}"

# ── 6. Download DUSt3R 512 checkpoint if missing ─────────────────────────────
CKPT_DIR="${PROJECT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"
CKPT_512="${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "${CKPT_512}" ]; then
  echo ""
  echo ">>> Downloading DUSt3R 512 checkpoint ..."
  wget -q --show-progress \
    https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -O "${CKPT_512}"
else
  echo "[skip] ${CKPT_512} already exists"
fi

echo ""
echo "================================================================"
echo "Preprocessing complete!  $(date)"
echo "  CO3D_ROOT : ${OUTPUT_DIR}"
echo "  BLUR_ROOT : ${DEGRADED_ROOT}"
echo "  DUST3R_CKPT: ${CKPT_512}"
echo ""
echo "Now submit training:"
echo "  sbatch finetune_blur/deblurdinat/train_dust3r_deblur_asds.sh"
echo "================================================================"
