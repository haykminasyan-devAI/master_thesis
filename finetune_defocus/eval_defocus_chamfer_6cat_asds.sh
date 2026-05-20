#!/usr/bin/env bash
# SLURM: Chamfer evaluation for IFAN+DUSt3R defocus — 6 held-out categories.
# Three pipelines: DUSt3R only | pretrained IFAN+DUSt3R | finetuned IFAN+DUSt3R
#
# Usage:
#   cd /home/asds/project_Hayk_Minasyan
#   sbatch finetune_defocus/eval_defocus_chamfer_6cat_asds.sh
#
# Overrides (sbatch --export=ALL,...):
#   CO3D_ROOT, DUST3R_CKPT, IFAN_REPO, IFAN_CKPT, FINETUNED_CKPT,
#   DEFOCUS_RADIUS, IMAGE_SIZE, N_FRAMES, OUT_ROOT
#
#SBATCH --job-name=eval_ifan_defocus_cd
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/eval_ifan_defocus_chamfer_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/eval_ifan_defocus_chamfer_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asds/project_Hayk_Minasyan}"
CO3D_ROOT="${CO3D_ROOT:-${PROJECT_DIR}/data/co3d_processed_10cat}"
# Raw CO3D tree (images/); processed dir often only has pointcloud.ply symlinks.
CO3D_RAW_ROOT="${CO3D_RAW_ROOT:-${PROJECT_DIR}/data/co3d}"
DUST3R_DIR="${DUST3R_DIR:-${PROJECT_DIR}/dust3r}"
DUST3R_CKPT="${DUST3R_CKPT:-${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
IFAN_REPO="${IFAN_REPO:-${PROJECT_DIR}/IFAN}"
IFAN_CKPT="${IFAN_CKPT:-${PROJECT_DIR}/IFAN/ckpt/IFAN.pytorch}"
FINETUNED_CKPT="${FINETUNED_CKPT:-${PROJECT_DIR}/finetune_defocus_runs/ifan_front_dust3r_freeze_train10_asds/checkpoint-best-val.pth}"

DEFOCUS_RADIUS="${DEFOCUS_RADIUS:-6}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
N_FRAMES="${N_FRAMES:-20}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/finetune_defocus_runs/ifan_front_dust3r_freeze_train10_asds/eval_chamfer_6cat_${SLURM_JOB_ID:-local}}"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
PYTHON="$(command -v python3)"
mkdir -p "${PROJECT_DIR}/logs" "${OUT_ROOT}"
export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

# Same 6 eval sequences as evaluation_blur_and_noise/sequences_6eval.inc.sh
declare -A SEQ=(
  [bottle]="34_1397_4376"
  [cup]="12_100_593"
  [donut]="110_13050_22740"
  [teddybear]="101_11758_21048"
  [couch]="105_12576_23188"
  [toytrain]="104_12352_22039"
)
ORDER=(bottle cup donut teddybear couch toytrain)
PIPELINES=(dust3r ifan_pretrained ifan_finetuned)

echo "=================================================================="
echo "Defocus Chamfer eval | $(date) | $(hostname)"
echo "CO3D_ROOT        : ${CO3D_ROOT}"
echo "CO3D_RAW_ROOT    : ${CO3D_RAW_ROOT}"
echo "DUST3R_CKPT      : ${DUST3R_CKPT}"
echo "IFAN_CKPT        : ${IFAN_CKPT}"
echo "FINETUNED_CKPT   : ${FINETUNED_CKPT}"
echo "DEFOCUS_RADIUS   : ${DEFOCUS_RADIUS}"
echo "OUT_ROOT         : ${OUT_ROOT}"
echo "=================================================================="

RESULTS_JSON="${OUT_ROOT}/per_run_results.jsonl"
: > "${RESULTS_JSON}"

for cat in "${ORDER[@]}"; do
  sid="${SEQ[$cat]}"
  for pl in "${PIPELINES[@]}"; do
    odir="${OUT_ROOT}/${pl}/${cat}_${sid}"
    mkdir -p "${odir}"
    echo ">>> ${pl}  ${cat}/${sid}"
    extra=()
    if [[ "${pl}" != "dust3r" ]]; then
      extra+=(--ifan_repo "${IFAN_REPO}" --ifan_ckpt "${IFAN_CKPT}")
    fi
    if [[ "${pl}" == "ifan_finetuned" ]]; then
      extra+=(--finetuned_weights "${FINETUNED_CKPT}")
    fi
    "${PYTHON}" "${PROJECT_DIR}/finetune_defocus/run_inference_defocus_chamfer.py" \
      --co3d_root "${CO3D_ROOT}" \
      --co3d_raw_root "${CO3D_RAW_ROOT}" \
      --category "${cat}" \
      --seq_id "${sid}" \
      --defocus_radius "${DEFOCUS_RADIUS}" \
      --dust3r_dir "${DUST3R_DIR}" \
      --dust3r_ckpt "${DUST3R_CKPT}" \
      --output_dir "${odir}" \
      --image_size "${IMAGE_SIZE}" \
      --n_frames "${N_FRAMES}" \
      --device cuda \
      --pipeline "${pl}" \
      --quiet \
      "${extra[@]}"

    cd_val="$(grep -E '^chamfer_distance:' "${odir}/metrics.txt" | head -1 | awk '{print $2}')"
    echo "{\"category\":\"${cat}\",\"seq_id\":\"${sid}\",\"pipeline\":\"${pl}\",\"chamfer_distance\":${cd_val}}" >> "${RESULTS_JSON}"
  done
done

"${PYTHON}" - <<PY
import json
from pathlib import Path
from statistics import mean

root = Path(r"${OUT_ROOT}")
lines = [
    json.loads(l)
    for l in (root / "per_run_results.jsonl").read_text().splitlines()
    if l.strip()
]
by_pl = {}
for r in lines:
    by_pl.setdefault(r["pipeline"], []).append(float(r["chamfer_distance"]))

order_cat = ["bottle", "cup", "donut", "teddybear", "couch", "toytrain"]
pl_order = ["dust3r", "ifan_pretrained", "ifan_finetuned"]

rows = []
for c in order_cat:
    row = [c]
    for pl in pl_order:
        v = next(
            (
                float(r["chamfer_distance"])
                for r in lines
                if r["category"] == c and r["pipeline"] == pl
            ),
            None,
        )
        row.append(f"{v:.8f}" if v is not None else "NA")
    rows.append(row)

means = [mean(by_pl[p]) for p in pl_order]
summary = (
    "| category | DUSt3R | pretrained_IFAN+DUSt3R | finetuned_IFAN+DUSt3R |\n"
)
summary += "|---|---:|---:|---:|\n"
for row in rows:
    summary += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n"
summary += (
    f"| **mean (6 cat)** | **{means[0]:.8f}** | **{means[1]:.8f}** | **{means[2]:.8f}** |\n"
)

out_md = root / "chamfer_table.md"
out_md.write_text(summary)
print(summary)
print(f"Wrote: {out_md}")
PY

echo "Done: $(date)"
