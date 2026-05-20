#!/usr/bin/env bash
# YSU: Chamfer evaluation for IFAN+DUSt3R defocus on 6 held-out categories,
# at three defocus radii (3, 6, 9). Three pipelines per radius:
#   dust3r | ifan_pretrained | ifan_finetuned
# Produces one Markdown sub-table per radius + a combined summary table.
#
# Usage:
#   cd ~/project_Hayk_Minasyan
#   sbatch finetune_defocus/eval_defocus_chamfer_6cat_ysu_multiradius.sh
#
# Optional overrides:
#   sbatch --export=ALL,FINETUNED_CKPT=/path/to/checkpoint-best-val.pth,RADII="3 6 9" \
#          finetune_defocus/eval_defocus_chamfer_6cat_ysu_multiradius.sh

#SBATCH --job-name=eval_ifan_defocus_cd_3r
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/hminasyan/project_Hayk_Minasyan/logs/eval_ifan_defocus_chamfer_3r_%j.log
#SBATCH --error=/home/hminasyan/project_Hayk_Minasyan/logs/eval_ifan_defocus_chamfer_3r_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_Hayk_Minasyan}"
CO3D_ROOT="${CO3D_ROOT:-/mnt/weka/hminasyan/data/co3d_processed}"
CO3D_RAW_ROOT="${CO3D_RAW_ROOT:-/mnt/weka/hminasyan/data/co3d}"
DUST3R_DIR="${DUST3R_DIR:-${PROJECT_DIR}/dust3r}"
DUST3R_CKPT="${DUST3R_CKPT:-/mnt/weka/hminasyan/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
IFAN_REPO="${IFAN_REPO:-${PROJECT_DIR}/IFAN}"
IFAN_CKPT="${IFAN_CKPT:-${PROJECT_DIR}/IFAN/ckpt/IFAN.pytorch}"
FINETUNED_CKPT="${FINETUNED_CKPT:-/mnt/weka/hminasyan/runs/finetune_defocus_runs/ifan_front_dust3r_seq8_randblur_ysu_4gpu_split6_2/checkpoint-best-val.pth}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
N_FRAMES="${N_FRAMES:-20}"
RADII="${RADII:-3 6 9}"
OUT_ROOT="${OUT_ROOT:-/mnt/weka/hminasyan/runs/finetune_defocus_runs/eval_chamfer_6cat_3radius_${SLURM_JOB_ID:-local}}"
export OUT_ROOT

VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/weka/hminasyan/co3d_env/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate co3d_env
else
  echo "ERROR: cannot activate co3d_env (no VENV_ACTIVATE, no Miniforge)" >&2
  exit 1
fi

PYTHON="$(command -v python3)"
mkdir -p "${PROJECT_DIR}/logs" "${OUT_ROOT}"
export PYTHONPATH="${PROJECT_DIR}/dust3r:${PROJECT_DIR}:${PYTHONPATH:-}"

# 1 alphabetically-first sequence per category (matches what's in co3d_processed/<cat>/).
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
echo "Defocus Chamfer eval (multi-radius)  | $(date) | $(hostname)"
echo "CO3D_ROOT       : ${CO3D_ROOT}"
echo "CO3D_RAW_ROOT   : ${CO3D_RAW_ROOT}"
echo "DUST3R_CKPT     : ${DUST3R_CKPT}"
echo "IFAN_CKPT       : ${IFAN_CKPT}"
echo "FINETUNED_CKPT  : ${FINETUNED_CKPT}"
echo "RADII           : ${RADII}"
echo "OUT_ROOT        : ${OUT_ROOT}"
echo "=================================================================="

if [[ ! -f "${FINETUNED_CKPT}" ]]; then
  echo "ERROR: FINETUNED_CKPT does not exist: ${FINETUNED_CKPT}" >&2
  echo "       Wait for training to produce checkpoint-best-val.pth, or override via --export=ALL,FINETUNED_CKPT=..." >&2
  exit 2
fi

GLOBAL_JSONL="${OUT_ROOT}/all_runs.jsonl"
: > "${GLOBAL_JSONL}"

for r in ${RADII}; do
  R_OUT="${OUT_ROOT}/radius_${r}"
  RES_JSON="${R_OUT}/per_run_results.jsonl"
  mkdir -p "${R_OUT}"
  : > "${RES_JSON}"

  echo
  echo "###################  RADIUS = ${r}  ###################"

  for cat in "${ORDER[@]}"; do
    sid="${SEQ[$cat]}"
    for pl in "${PIPELINES[@]}"; do
      odir="${R_OUT}/${pl}/${cat}_${sid}"
      mkdir -p "${odir}"
      echo ">>> r=${r}  ${pl}  ${cat}/${sid}"
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
        --defocus_radius "${r}" \
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
      echo "{\"radius\":${r},\"category\":\"${cat}\",\"seq_id\":\"${sid}\",\"pipeline\":\"${pl}\",\"chamfer_distance\":${cd_val}}" \
        | tee -a "${RES_JSON}" >> "${GLOBAL_JSONL}"
    done
  done
done

# Build per-radius and combined Markdown tables.
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
from statistics import mean

root = Path(os.environ["OUT_ROOT"])
all_runs = [json.loads(l) for l in (root/"all_runs.jsonl").read_text().splitlines() if l.strip()]

order_cat = ["bottle", "cup", "donut", "teddybear", "couch", "toytrain"]
pl_order = ["dust3r", "ifan_pretrained", "ifan_finetuned"]
pl_label = {
    "dust3r": "DUSt3R",
    "ifan_pretrained": "pretrained_IFAN+DUSt3R",
    "ifan_finetuned": "finetuned_IFAN+DUSt3R",
}

radii = sorted({r["radius"] for r in all_runs})

def md_table_for_radius(R):
    rows = []
    for c in order_cat:
        row = [c]
        for pl in pl_order:
            v = next(
                (float(r["chamfer_distance"]) for r in all_runs
                 if r["radius"]==R and r["category"]==c and r["pipeline"]==pl),
                None,
            )
            row.append(f"{v:.8f}" if v is not None else "NA")
        rows.append(row)
    means = [mean([float(r["chamfer_distance"]) for r in all_runs
                   if r["radius"]==R and r["pipeline"]==pl]) for pl in pl_order]
    s  = f"### Defocus radius = {R}\n\n"
    s += "| category | DUSt3R | pretrained_IFAN+DUSt3R | finetuned_IFAN+DUSt3R |\n"
    s += "|---|---:|---:|---:|\n"
    for row in rows:
        s += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n"
    s += f"| **mean (6 cat)** | **{means[0]:.8f}** | **{means[1]:.8f}** | **{means[2]:.8f}** |\n\n"
    return s

# Combined wide table: rows = category, cols = (radius x pipeline).
def md_combined():
    head = "| category | " + " | ".join(
        [f"r={R} {pl_label[pl]}" for R in radii for pl in pl_order]
    ) + " |\n"
    sep  = "|---|" + "---:|"*len(radii)*len(pl_order) + "\n"
    body = ""
    for c in order_cat:
        cells = [c]
        for R in radii:
            for pl in pl_order:
                v = next((float(r["chamfer_distance"]) for r in all_runs
                          if r["radius"]==R and r["category"]==c and r["pipeline"]==pl), None)
                cells.append(f"{v:.6f}" if v is not None else "NA")
        body += "| " + " | ".join(cells) + " |\n"
    means = []
    for R in radii:
        for pl in pl_order:
            vals = [float(r["chamfer_distance"]) for r in all_runs
                    if r["radius"]==R and r["pipeline"]==pl]
            means.append(mean(vals) if vals else float("nan"))
    body += "| **mean** | " + " | ".join(f"**{m:.6f}**" for m in means) + " |\n"
    return "## Combined (6 cats × radii × pipelines)\n\n" + head + sep + body + "\n"

doc = "# Defocus Chamfer evaluation — 6 held-out categories, 3 radii\n\n"
for R in radii:
    doc += md_table_for_radius(R)
doc += md_combined()

(out_md := root/"chamfer_table_3radius.md").write_text(doc)
print(doc)
print(f"Wrote: {out_md}")
PY

echo
echo "Done: $(date)"
echo "Markdown table -> ${OUT_ROOT}/chamfer_table_3radius.md"
