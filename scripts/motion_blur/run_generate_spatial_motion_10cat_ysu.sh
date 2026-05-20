#!/usr/bin/env bash
# Generate spatially varying motion-blur dataset for 10 categories on YSU.
# Output layout:
#   /mnt/weka/hminasyan/outputs/degraded_frames_motion_10cat/<cat>/<seq>/spatial_patchwise_g8_k25_61_seed123/*.jpg
#
# Submit:
#   cd /home/hminasyan/project_Hayk_Minasyan
#   sbatch scripts/motion_blur/run_generate_spatial_motion_10cat_ysu.sh

#SBATCH --job-name=motion_spatial_10cat
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/motion_spatial_10cat_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/motion_spatial_10cat_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/hminasyan/project_Hayk_Minasyan}"
: "${CO3D_ROOT:=/mnt/weka/hminasyan/data/co3d}"
: "${OUT_ROOT:=/mnt/weka/hminasyan/outputs/degraded_frames_motion_10cat}"
: "${SEQ_JSON:=${PROJECT_DIR}/finetune_blur/sequences_10cat.json}"
: "${SCRIPT_PATH:=${PROJECT_DIR}/scripts/motion_blur/generate_motion_blur_examples.py}"

: "${GRID_SIZE:=8}"
: "${PATCH_MIN_KERNEL:=25}"
: "${PATCH_MAX_KERNEL:=61}"
: "${SEED:=123}"
: "${TAG:=spatial_patchwise_g8_k25_61_seed123}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_ROOT}"
cd "${PROJECT_DIR}"

if [[ -f "/mnt/weka/hminasyan/co3d_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/mnt/weka/hminasyan/co3d_env/bin/activate"
fi

PYTHON="${PYTHON:-python3}"

echo "================================================================"
echo "Generate 10-cat spatial motion blur | $(date) | $(hostname)"
echo "PROJECT_DIR      : ${PROJECT_DIR}"
echo "CO3D_ROOT        : ${CO3D_ROOT}"
echo "OUT_ROOT         : ${OUT_ROOT}"
echo "SEQ_JSON         : ${SEQ_JSON}"
echo "SCRIPT_PATH      : ${SCRIPT_PATH}"
echo "GRID_SIZE        : ${GRID_SIZE}"
echo "PATCH_MIN_KERNEL : ${PATCH_MIN_KERNEL}"
echo "PATCH_MAX_KERNEL : ${PATCH_MAX_KERNEL}"
echo "SEED             : ${SEED}"
echo "TAG              : ${TAG}"
echo "================================================================"

[[ -f "${SCRIPT_PATH}" ]] || { echo "ERROR: missing script: ${SCRIPT_PATH}"; exit 1; }
[[ -f "${SEQ_JSON}" ]] || { echo "ERROR: missing sequence json: ${SEQ_JSON}"; exit 1; }
[[ -d "${CO3D_ROOT}" ]] || { echo "ERROR: missing CO3D root: ${CO3D_ROOT}"; exit 1; }

"${PYTHON}" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

project = Path(os.environ.get("PROJECT_DIR", "/home/hminasyan/project_Hayk_Minasyan"))
co3d_root = Path(os.environ.get("CO3D_ROOT", "/mnt/weka/hminasyan/data/co3d"))
out_root = Path(os.environ.get("OUT_ROOT", "/mnt/weka/hminasyan/outputs/degraded_frames_motion_10cat"))
seq_json = Path(os.environ.get("SEQ_JSON", str(project / "finetune_blur/sequences_10cat.json")))
script = Path(os.environ.get("SCRIPT_PATH", str(project / "scripts/motion_blur/generate_motion_blur_examples.py")))

grid_size = os.environ.get("GRID_SIZE", "8")
patch_min_kernel = os.environ.get("PATCH_MIN_KERNEL", "25")
patch_max_kernel = os.environ.get("PATCH_MAX_KERNEL", "61")
seed = os.environ.get("SEED", "123")
tag = os.environ.get("TAG", "spatial_patchwise_g8_k25_61_seed123")
python_bin = os.environ.get("PYTHON", "python3")

extras = [
    "original.png",
    "linear_horizontal_L31.png",
    "linear_vertical_L31.png",
    "angle_15deg_L31.png",
    "angle_30deg_L31.png",
    "angle_45deg_L31.png",
    "angle_60deg_L31.png",
    "angle_75deg_L31.png",
    "camera_shake_len20_shift5_seed42.png",
    "camera_shake_len20_shift5_seed123.png",
]

seqs = json.loads(seq_json.read_text())

for cat, sid in sorted(seqs.items()):
    img_dir = co3d_root / cat / sid / "images"
    out_dir = out_root / cat / sid / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        print(f"MISSING images: {img_dir}")
        continue

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not imgs:
        print(f"NO images: {img_dir}")
        continue

    wrote = 0
    for p in imgs:
        dst = out_dir / p.name
        if dst.exists():
            continue

        subprocess.check_call([
            python_bin, str(script),
            "--image_path", str(p),
            "--output_dir", str(out_dir),
            "--seed", str(seed),
            "--grid_size", str(grid_size),
            "--patch_min_kernel", str(patch_min_kernel),
            "--patch_max_kernel", str(patch_max_kernel),
        ])

        gen = out_dir / tag
        gen_png = Path(f"{gen}.png")
        if gen_png.exists():
            gen_png.replace(dst)
            wrote += 1

        for name in extras:
            ep = out_dir / name
            if ep.exists():
                ep.unlink()

    total = len(list(out_dir.glob("*.jpg")))
    print(f"[ok] {cat}/{sid}: new={wrote}, total_jpg={total}")
PY

echo "Done: $(date)"
