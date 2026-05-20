#!/usr/bin/env bash
# Run DUSt3R preprocessing for the 6 CO3D experiment sequences on YSU.
# This is a ONE-TIME setup step — run interactively BEFORE submitting train_blur.sh.
#
# Usage:
#   bash finetune_blur/preprocess_for_training.sh
#
# Override defaults:
#   CO3D_DIR=/path/to/raw/co3d  PYTHON=/path/to/python  bash finetune_blur/preprocess_for_training.sh

set -euo pipefail

CO3D_DIR="${CO3D_DIR:-/mnt/weka/hminasyan/data/co3d}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/weka/hminasyan/data/co3d_processed}"
PYTHON="${PYTHON:-/mnt/weka/hminasyan/co3d_env/bin/python3}"
CKPT_DIR="${CKPT_DIR:-/mnt/weka/hminasyan/checkpoints}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DUST3R_DIR="${PROJECT_DIR}/dust3r"

CATEGORIES=(teddybear hydrant cup bottle toybus toytrain)

# Both sequences per category
declare -A SEQ1 SEQ2
SEQ1=(
    [teddybear]="101_11758_21048"
    [hydrant]="106_12648_23157"
    [cup]="12_100_593"
    [bottle]="34_1397_4376"
    [toybus]="111_13154_25988"
    [toytrain]="104_12352_22039"
)
SEQ2=(
    [teddybear]="101_11763_21624"
    [hydrant]="106_12653_23216"
    [cup]="14_158_900"
    [bottle]="34_1402_4474"
    [toybus]="104_12348_21852"
    [toytrain]="111_13149_23190"
)

BLUR_SIGMAS=(5 10 20)
DEGRADED_ROOT="${DEGRADED_ROOT:-/mnt/weka/hminasyan/outputs/degraded_frames}"

echo "================================================================"
echo "CO3D raw     : ${CO3D_DIR}"
echo "CO3D output  : ${OUTPUT_DIR}"
echo "Degraded root: ${DEGRADED_ROOT}"
echo "Python       : ${PYTHON}"
echo "================================================================"

# ── Step 0: Bootstrap missing CO3D set_lists (needed by DUSt3R preprocessor) ─
echo ""
echo ">>> Checking CO3D set_lists ..."
for CAT in "${CATEGORIES[@]}"; do
    SET_LIST_DIR="${CO3D_DIR}/${CAT}/set_lists"
    if [ -d "${SET_LIST_DIR}" ] && [ "$(ls -A "${SET_LIST_DIR}" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [ok] ${CAT}/set_lists exists"
        continue
    fi

    echo "  [gen] ${CAT}/set_lists is missing -> creating minimal fewview_train_custom.json"
    mkdir -p "${SET_LIST_DIR}"

    "${PYTHON}" - <<PY
import gzip
import json
import os

cat = "${CAT}"
co3d_dir = "${CO3D_DIR}"
seq1 = "${SEQ1[$CAT]}"
seq2 = "${SEQ2[$CAT]}"
keep = {seq1, seq2}

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

# ── Step 1: Preprocess CO3D (images, depths, masks, .npz) ─────────────────────
mkdir -p "${OUTPUT_DIR}"

cd "${DUST3R_DIR}/datasets_preprocess"
for CAT in "${CATEGORIES[@]}"; do
    DONE_FLAG="${OUTPUT_DIR}/${CAT}/selected_seqs_train.json"
    if [ -f "${DONE_FLAG}" ]; then
        echo "[skip] ${CAT} already preprocessed"
    else
        echo ">>> Preprocessing ${CAT} ..."
        # DUSt3R skips ALL categories if these root files exist (leftover from a
        # previous category run). Remove so this --category run actually executes.
        rm -f "${OUTPUT_DIR}/selected_seqs_train.json" "${OUTPUT_DIR}/selected_seqs_test.json"
        "${PYTHON}" preprocess_co3d.py \
            --co3d_dir "${CO3D_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --category "${CAT}" \
            --num_sequences_per_object 50 \
            --seed 42
    fi
done

# ── Step 2: Generate blurred frames for seq2 (from preprocessed crops) ────────
# Blur must match OUTPUT_DIR image size so Co3dBlur RGB aligns with depth masks.
echo ""
echo ">>> Generating blurred frames for second sequences (preprocessed images) ..."
cd "${PROJECT_DIR}"
for CAT in "${CATEGORIES[@]}"; do
    S2="${SEQ2[$CAT]}"
    PROC_IMGS="${OUTPUT_DIR}/${CAT}/${S2}/images"
    for SIGMA in "${BLUR_SIGMAS[@]}"; do
        OUT="${DEGRADED_ROOT}/${CAT}/${S2}/blur_s${SIGMA}"
        if [ -d "${OUT}" ] && [ "$(ls -A "${OUT}" 2>/dev/null | wc -l)" -gt 0 ] && [ "${FORCE_REGEN_BLUR:-0}" != "1" ]; then
            echo "  [skip] ${CAT}/${S2}/blur_s${SIGMA}"
        else
            echo "  [gen]  ${CAT}/${S2}/blur_s${SIGMA} ..."
            mkdir -p "${OUT}"
            "${PYTHON}" "${SCRIPT_DIR}/degrade_blur.py" \
                --images_dir "${PROC_IMGS}" \
                --output_dir "${OUT}" \
                --blur_sigma "${SIGMA}"
        fi
    done
done

# ── Step 3: Create 12-sequence selected_seqs_*.json ───────────────────────────
echo ""
echo ">>> Creating selected_seqs JSON for all 12 sequences ..."
export PYTHONPATH="${DUST3R_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
"${PYTHON}" "${SCRIPT_DIR}/make_selected_seqs.py" \
    --co3d_processed "${OUTPUT_DIR}"

# ── Step 4: Download pretrained DUSt3R checkpoint (224 linear) ────────────────
mkdir -p "${CKPT_DIR}"
CKPT_224="${CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
if [ ! -f "${CKPT_224}" ]; then
    echo ""
    echo ">>> Downloading DUSt3R_ViTLarge_BaseDecoder_224_linear.pth ..."
    wget -q --show-progress \
        https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \
        -O "${CKPT_224}"
else
    echo "[skip] checkpoint already at ${CKPT_224}"
fi

echo ""
echo "================================================================"
echo "Setup complete!"
echo "  CO3D_ROOT : ${OUTPUT_DIR}"
echo "  BLUR_ROOT : ${DEGRADED_ROOT}"
echo "  PRETRAINED: ${CKPT_224}"
echo ""
echo "Submit training with:"
echo "  sbatch finetune_blur/train_blur.sh"
echo "================================================================"
