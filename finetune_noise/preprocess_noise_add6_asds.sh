#!/usr/bin/env bash
#SBATCH --job-name=co3d_add6_noise
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/co3d_add6_noise_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/co3d_add6_noise_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
PYTHON="${PYTHON:-python3}"

CO3D_RAW="${CO3D_RAW:-${PROJECT_DIR}/data/co3d}"
CO3D_PROCESSED="${CO3D_PROCESSED:-${PROJECT_DIR}/data/co3d_processed_10cat}"
NOISE_ROOT="${NOISE_ROOT:-${PROJECT_DIR}/outputs/noisy_frames_10cat}"
NOISE_SIGMAS=(${NOISE_SIGMAS:-70 80})

# Keep eval-6 untouched; extend train pool with these 6.
CATEGORIES=(hydrant laptop motorcycle parkingmeter skateboard toybus)

echo "============================================================"
echo "Add 6 categories (processed + noise) | $(date) | $(hostname)"
echo "CO3D raw       : ${CO3D_RAW}"
echo "CO3D processed : ${CO3D_PROCESSED}"
echo "NOISE root     : ${NOISE_ROOT}"
echo "Noise sigmas   : ${NOISE_SIGMAS[*]}"
echo "Categories     : ${CATEGORIES[*]}"
echo "============================================================"

mkdir -p "${CO3D_PROCESSED}" "${NOISE_ROOT}"

# 1) Build minimal fewview_train_custom.json from all frames if set_lists missing.
for CAT in "${CATEGORIES[@]}"; do
  CAT_DIR="${CO3D_RAW}/${CAT}"
  [[ -d "${CAT_DIR}" ]] || { echo "ERROR: missing category dir ${CAT_DIR}"; exit 1; }
  [[ -f "${CAT_DIR}/frame_annotations.jgz" ]] || {
    echo "ERROR: missing ${CAT_DIR}/frame_annotations.jgz (download category first)"; exit 1;
  }
  mkdir -p "${CAT_DIR}/set_lists"
  OUT_SETLIST="${CAT_DIR}/set_lists/fewview_train_custom.json"
  if [[ -f "${OUT_SETLIST}" ]]; then
    echo "[skip] ${OUT_SETLIST}"
  else
    echo "[gen ] ${OUT_SETLIST}"
    "${PYTHON}" - <<PY
import gzip, json, os
cat_dir = "${CAT_DIR}"
frame_file = os.path.join(cat_dir, "frame_annotations.jgz")
with gzip.open(frame_file, "r") as fin:
    frame_data = json.loads(fin.read())
entries = []
for f in frame_data:
    seq_name = f.get("sequence_name")
    frame_number = f.get("frame_number")
    image_path = f.get("image", {}).get("path")
    if seq_name is None or frame_number is None or image_path is None:
        continue
    entries.append([seq_name, frame_number, image_path])
entries.sort(key=lambda x: (x[0], x[1]))
out = os.path.join(cat_dir, "set_lists", "fewview_train_custom.json")
with open(out, "w") as fw:
    json.dump({"train": entries, "test": entries}, fw)
print(f"wrote {out} with {len(entries)} entries")
PY
  fi
done

# 2) Preprocess each added category into the shared processed root.
cd "${PROJECT_DIR}/dust3r/datasets_preprocess"
for CAT in "${CATEGORIES[@]}"; do
  echo ""
  echo ">>> Preprocess ${CAT}"
  rm -f "${CO3D_PROCESSED}/selected_seqs_train.json" "${CO3D_PROCESSED}/selected_seqs_test.json"
  "${PYTHON}" preprocess_co3d.py \
    --co3d_dir "${CO3D_RAW}" \
    --output_dir "${CO3D_PROCESSED}" \
    --category "${CAT}" \
    --num_sequences_per_object 50 \
    --seed 42
done

# 3) Generate noise from preprocessed images for selected sequence of each category.
cd "${PROJECT_DIR}"
for CAT in "${CATEGORIES[@]}"; do
  SEL_JSON="${CO3D_PROCESSED}/${CAT}/selected_seqs_train.json"
  [[ -f "${SEL_JSON}" ]] || { echo "ERROR: missing ${SEL_JSON}"; exit 1; }
  SID="$("${PYTHON}" -c "import json; d=json.load(open('${SEL_JSON}')); print(next(iter(d.get('${CAT}',{}).keys()), ''))")"
  [[ -n "${SID}" ]] || { echo "ERROR: could not read selected seq for ${CAT}"; exit 1; }
  SRC_IMGS="${CO3D_PROCESSED}/${CAT}/${SID}/images"
  [[ -d "${SRC_IMGS}" ]] || { echo "ERROR: missing images ${SRC_IMGS}"; exit 1; }
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

# 4) Rebuild root selected_seqs_{train,test}.json from all category subfolders.
"${PYTHON}" - <<PY
import json, os, glob
root = "${CO3D_PROCESSED}"
all_train = {}
for path in glob.glob(os.path.join(root, "*", "selected_seqs_train.json")):
    data = json.load(open(path))
    all_train.update(data)
for split in ("train", "test"):
    out = os.path.join(root, f"selected_seqs_{split}.json")
    with open(out, "w") as f:
        json.dump(all_train, f, indent=2, sort_keys=True)
    print(f"wrote {out} with {len(all_train)} categories")
PY

echo ""
echo "Done: $(date)"
echo "Processed root: ${CO3D_PROCESSED}"
echo "Noise root    : ${NOISE_ROOT}"
