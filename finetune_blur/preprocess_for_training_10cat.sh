#!/usr/bin/env bash
# One-time preprocessing for 10 CO3D categories (one sequence each) before 512 training.
#
# Categories: apple, banana, baseballbat, baseballglove, bicycle, bowl, broccoli, cake, car, carrot
#
# Usage:
#   bash finetune_blur/preprocess_for_training_10cat.sh
#
# Requires raw CO3D for: apple, banana, baseballbat, baseballglove, bicycle, bowl,
# broccoli, cake, car, carrot under CO3D_DIR. If discover fails, download first, e.g.:
#   cd <repo>/co3d/co3d && python download_dataset.py \
#     --download_folder /mnt/weka/hminasyan/data/co3d \
#     --download_categories apple,banana,baseballbat,baseballglove,bicycle,bowl,broccoli,cake,car,carrot
#
# Env overrides:
#   CO3D_DIR=/mnt/weka/hminasyan/data/co3d
#   OUTPUT_DIR=/mnt/weka/hminasyan/data/co3d_processed_10cat
#   DEGRADED_ROOT=/mnt/weka/hminasyan/outputs/degraded_frames_10cat
#   PYTHON=/mnt/weka/hminasyan/co3d_env/bin/python3

set -euo pipefail

CO3D_DIR="${CO3D_DIR:-/mnt/weka/hminasyan/data/co3d}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/weka/hminasyan/data/co3d_processed_10cat}"
PYTHON="${PYTHON:-/mnt/weka/hminasyan/co3d_env/bin/python3}"
CKPT_DIR="${CKPT_DIR:-/mnt/weka/hminasyan/checkpoints}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DUST3R_DIR="${PROJECT_DIR}/dust3r"

CATEGORIES=(apple banana baseballbat baseballglove bicycle bowl broccoli cake car carrot)
BLUR_SIGMAS=(5 10 20)
DEGRADED_ROOT="${DEGRADED_ROOT:-/mnt/weka/hminasyan/outputs/degraded_frames_10cat}"

SEQ_JSON="${SEQ_JSON:-${SCRIPT_DIR}/sequences_10cat.json}"

echo "================================================================"
echo "CO3D raw     : ${CO3D_DIR}"
echo "CO3D output  : ${OUTPUT_DIR}"
echo "Degraded root: ${DEGRADED_ROOT}"
echo "Sequences    : ${SEQ_JSON}"
echo "Python       : ${PYTHON}"
echo "================================================================"

# ── one sequence per category: reuse JSON or auto-discover ───────────────────
echo ""
if [ -f "${SEQ_JSON}" ]; then
  echo ">>> Using existing sequence manifest: ${SEQ_JSON}"
else
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
  echo "  ${CAT} -> ${SEQ_ID[$CAT]}"
done

# ── set_lists for DUSt3R preprocessor (always fewview_train_custom for our seq) ─
echo ""
echo ">>> Writing set_lists/fewview_train_custom.json per category ..."
for CAT in "${CATEGORIES[@]}"; do
  SET_LIST_DIR="${CO3D_DIR}/${CAT}/set_lists"
  SID="${SEQ_ID[$CAT]}"
  mkdir -p "${SET_LIST_DIR}"

  echo "  [gen] ${CAT}/set_lists -> fewview_train_custom.json (${SID})"

  "${PYTHON}" - <<PY
import gzip
import json
import os

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
print(f"    wrote {out} with {len(entries)} entries")
PY
done

# ── Preprocess CO3D (512-friendly img_size in preprocess_co3d defaults) ─────
mkdir -p "${OUTPUT_DIR}"
cd "${DUST3R_DIR}/datasets_preprocess"

for CAT in "${CATEGORIES[@]}"; do
  DONE_FLAG="${OUTPUT_DIR}/${CAT}/selected_seqs_train.json"
  if [ -f "${DONE_FLAG}" ]; then
    echo "[skip] ${CAT} already preprocessed"
  else
    echo ">>> Preprocessing ${CAT} ..."
    rm -f "${OUTPUT_DIR}/selected_seqs_train.json" "${OUTPUT_DIR}/selected_seqs_test.json"
    "${PYTHON}" preprocess_co3d.py \
      --co3d_dir "${CO3D_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --category "${CAT}" \
      --num_sequences_per_object 50 \
      --seed 42
  fi
done

# ── Blurred frames for each sequence ─────────────────────────────────────────
# Blur must be applied to PREPROCESSED crops (OUTPUT_DIR) so RGB matches depth
# in co3d_processed. Blurring raw CO3D images causes size mismatch in train_blur.
echo ""
echo ">>> Generating blurred frames (from preprocessed images) ..."
cd "${PROJECT_DIR}"
for CAT in "${CATEGORIES[@]}"; do
  SID="${SEQ_ID[$CAT]}"
  PROC_IMGS="${OUTPUT_DIR}/${CAT}/${SID}/images"
  if [ ! -d "${PROC_IMGS}" ]; then
    echo "  [WARN] ${CAT}/${SID}/images not found (preprocessing may have filtered this sequence) — skipping blur"
    continue
  fi
  for SIGMA in "${BLUR_SIGMAS[@]}"; do
    OUT="${DEGRADED_ROOT}/${CAT}/${SID}/blur_s${SIGMA}"
    if [ -d "${OUT}" ] && [ "$(ls -A "${OUT}" 2>/dev/null | wc -l)" -gt 0 ] && [ "${FORCE_REGEN_BLUR:-0}" != "1" ]; then
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

# ── selected_seqs JSON at OUTPUT_DIR root ─────────────────────────────────────
echo ""
echo ">>> Writing selected_seqs_*.json ..."
export PYTHONPATH="${DUST3R_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
"${PYTHON}" "${SCRIPT_DIR}/make_selected_seqs_10cat.py" \
  --co3d_processed "${OUTPUT_DIR}" \
  --sequences_json "${SEQ_JSON}"

# ── 512 checkpoint (for training) ────────────────────────────────────────────
mkdir -p "${CKPT_DIR}"
CKPT_512="${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "${CKPT_512}" ]; then
  echo ""
  echo ">>> Downloading DUSt3R 512 checkpoint ..."
  wget -q --show-progress \
    https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -O "${CKPT_512}"
else
  echo "[skip] ${CKPT_512}"
fi

echo ""
echo "================================================================"
echo "Setup complete for 10 categories."
echo "  CO3D_ROOT : ${OUTPUT_DIR}"
echo "  BLUR_ROOT : ${DEGRADED_ROOT}"
echo "  PRETRAINED: ${CKPT_512}"
echo ""
echo "Submit 512 training:"
echo "  sbatch finetune_blur/train_blur_512.sh"
echo "================================================================"
