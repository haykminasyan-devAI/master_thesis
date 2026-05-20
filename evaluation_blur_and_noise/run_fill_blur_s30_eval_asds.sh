#!/usr/bin/env bash
# One-shot ASDS job:
#  1) Generate missing blur_s30 folders for the 6 eval sequences under outputs/degraded_frames
#  2) Run blur=30-only eval pass
#  3) Merge blur=30 rows into the existing main CSV + regenerate summary
#
# Submit:
#   cd /home/asds/project_Hayk_Minasyan
#   sbatch evaluation_blur_and_noise/run_fill_blur_s30_eval_asds.sh

#SBATCH --job-name=eval_blur30_fill
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/eval_blur30_fill_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/eval_blur30_fill_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ -f /home/asds/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/asds/miniforge3/etc/profile.d/conda.sh
  conda activate co3d_env
fi
PYTHON="${PYTHON:-python3}"

# -------------------------
# Paths and sequence list
# -------------------------
BLUR_ROOT="${BLUR_ROOT:-${PROJECT_DIR}/outputs/degraded_frames}"
NOISE_ROOT="${NOISE_ROOT:-${PROJECT_DIR}/outputs/degraded_frames}"
RAW_CO3D="${RAW_CO3D:-${PROJECT_DIR}/data/co3d}"
TARGET_SIGMA=30

BASE_OUT_ROOT="${BASE_OUT_ROOT:-${PROJECT_DIR}/outputs/dust3r_eval_6seq_blur_noise_20260420}"
BASE_CSV="${BASE_OUT_ROOT}/chamfer_results.csv"
BASE_SUMMARY="${BASE_OUT_ROOT}/chamfer_results.summary.txt"

PATCH_OUT_ROOT="${PATCH_OUT_ROOT:-${PROJECT_DIR}/outputs/dust3r_eval_6seq_blur_noise_blur30_patch_$(date +%Y%m%d_%H%M%S)}"
PATCH_CSV="${PATCH_OUT_ROOT}/chamfer_results.csv"

echo "================================================================"
echo "blur_s30 fill + eval patch | $(date) | $(hostname)"
echo "BLUR_ROOT       : ${BLUR_ROOT}"
echo "NOISE_ROOT      : ${NOISE_ROOT}"
echo "RAW_CO3D        : ${RAW_CO3D}"
echo "BASE_OUT_ROOT   : ${BASE_OUT_ROOT}"
echo "PATCH_OUT_ROOT  : ${PATCH_OUT_ROOT}"
echo "================================================================"

# shellcheck source=/dev/null
source "${SEQUENCES_FILE:-${PROJECT_DIR}/evaluation_blur_and_noise/sequences_6eval.inc.sh}"

# ---------------------------------------
# 1) Generate blur_s30 from raw images
# ---------------------------------------
for CAT in "${!SEQUENCES[@]}"; do
  SID="${SEQUENCES[$CAT]}"
  SRC="${RAW_CO3D}/${CAT}/${SID}/images"
  OUT="${BLUR_ROOT}/${CAT}/${SID}/blur_s${TARGET_SIGMA}"
  if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: missing raw images dir: ${SRC}"
    exit 1
  fi
  if [[ -d "${OUT}" ]] && [[ "$(ls -A "${OUT}" 2>/dev/null | wc -l)" -gt 0 ]]; then
    echo "[skip] ${CAT}/${SID}/blur_s${TARGET_SIGMA} exists"
  else
    echo "[gen ] ${CAT}/${SID}/blur_s${TARGET_SIGMA}"
    mkdir -p "${OUT}"
    "${PYTHON}" "${PROJECT_DIR}/finetune_blur/degrade_blur.py" \
      --images_dir "${SRC}" \
      --output_dir "${OUT}" \
      --blur_sigma "${TARGET_SIGMA}"
  fi
done

# --------------------------------------------------
# 2) Run sigma-30-only blur eval into patch output
# --------------------------------------------------
echo ""
echo ">>> Running blur=30-only eval patch pass ..."
BLUR_ROOT="${BLUR_ROOT}" \
NOISE_ROOT="${NOISE_ROOT}" \
BLUR_EVAL_SIGMAS="30" \
NOISE_EVAL_SIGMAS="" \
OUT_ROOT="${PATCH_OUT_ROOT}" \
bash "${PROJECT_DIR}/evaluation_blur_and_noise/run_eval_6seq_blur_noise_chamfer.sh"

if [[ ! -f "${PATCH_CSV}" ]]; then
  echo "ERROR: patch CSV missing: ${PATCH_CSV}"
  exit 1
fi

if [[ ! -f "${BASE_CSV}" ]]; then
  echo "ERROR: base CSV missing (nothing to patch): ${BASE_CSV}"
  exit 1
fi

# --------------------------------------------------
# 3) Merge patched blur=30 rows into base CSV
# --------------------------------------------------
echo ""
echo ">>> Merging blur=30 rows into ${BASE_CSV} ..."
"${PYTHON}" - <<PY
import csv
from collections import defaultdict
from pathlib import Path

base_csv = Path("${BASE_CSV}")
patch_csv = Path("${PATCH_CSV}")
base_summary = Path("${BASE_SUMMARY}")

base_rows = list(csv.DictReader(base_csv.open()))
patch_rows = list(csv.DictReader(patch_csv.open()))

patch_key = {(r["kind"], r["sigma"], r["category"], r["seq_id"], r["pipeline"]): r for r in patch_rows}

merged = []
for r in base_rows:
    k = (r["kind"], r["sigma"], r["category"], r["seq_id"], r["pipeline"])
    if k in patch_key:
        merged.append(patch_key[k])
    else:
        merged.append(r)

existing_keys = {(r["kind"], r["sigma"], r["category"], r["seq_id"], r["pipeline"]) for r in merged}
for k, r in patch_key.items():
    if k not in existing_keys:
        merged.append(r)

fieldnames = ["kind", "sigma", "category", "seq_id", "pipeline", "chamfer_distance"]
with base_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in merged:
        w.writerow({k: r.get(k, "") for k in fieldnames})

# Recompute summary with canonical sigma ordering
acc = defaultdict(list)
for r in merged:
    try:
        v = float(r["chamfer_distance"])
    except (TypeError, ValueError):
        continue
    acc[(r["kind"], r["sigma"], r["pipeline"])].append(v)

lines = [str(base_csv), "=" * 72, "Mean Chamfer over sequences (lower is better)", ""]
blur_pl = ["dust3r", "deblur_pretrained", "deblur_finetuned"]
noise_pl = ["dust3r", "uformer_pretrained", "uformer_finetuned"]
for kind, sigmas, pls in (
    ("blur", ["5", "10", "20", "30"], blur_pl),
    ("noise", ["30", "50", "70"], noise_pl),
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
base_summary.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Wrote merged CSV: {base_csv}")
print(f"Wrote merged summary: {base_summary}")
PY

echo ""
echo "Done: $(date)"
echo "Merged results:"
echo "  ${BASE_CSV}"
echo "  ${BASE_SUMMARY}"
