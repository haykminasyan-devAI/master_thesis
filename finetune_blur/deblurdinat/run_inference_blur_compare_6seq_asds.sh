#!/usr/bin/env bash
#SBATCH --job-name=blur6seq_cmp
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/blur6seq_cmp_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/blur6seq_cmp_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
VENV_PYTHON="$(command -v python3)"

: "${SIGMAS:=5 10 20}"
: "${N_FRAMES:=20}"
: "${IMAGE_SIZE:=224}"

: "${DUST3R_DIR:=${PROJECT_DIR}/dust3r}"
: "${DUST3R_CKPT:=${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
: "${BEST_CKPT:=${PROJECT_DIR}/finetune_blur_runs/deblurdinat_dust3r_asds_224/joint_sigmas_5_10_20/checkpoint-best-val.pth}"

# These six are the standard multi-sequence set used in your experiments.
declare -A SEQUENCES
SEQUENCES["teddybear"]="101_11758_21048"
SEQUENCES["hydrant"]="106_12648_23157"
SEQUENCES["cup"]="12_100_593"
SEQUENCES["bottle"]="34_1397_4376"
SEQUENCES["toybus"]="111_13154_25988"
SEQUENCES["toytrain"]="104_12352_22039"

: "${BLUR_ROOT:=${PROJECT_DIR}/outputs/degraded_frames}"
: "${CO3D_ROOT:=${PROJECT_DIR}/data/co3d}"
: "${OUT_ROOT:=${PROJECT_DIR}/outputs/dust3r_eval_6seq}"

if [[ ! -f "${DUST3R_CKPT}" ]]; then
  echo "ERROR: Missing DUSt3R checkpoint: ${DUST3R_CKPT}"
  exit 1
fi
if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "ERROR: Missing finetuned checkpoint: ${BEST_CKPT}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "6-seq blur comparison job | $(date) | $(hostname)"
echo "Sigmas       : ${SIGMAS}"
echo "N_FRAMES     : ${N_FRAMES}"
echo "IMAGE_SIZE   : ${IMAGE_SIZE}"
echo "DUSt3R ckpt  : ${DUST3R_CKPT}"
echo "Best ckpt    : ${BEST_CKPT}"
echo "Blur root    : ${BLUR_ROOT}"
echo "CO3D root    : ${CO3D_ROOT}"
echo "Output root  : ${OUT_ROOT}"
echo "============================================================"

for CATEGORY in "${!SEQUENCES[@]}"; do
  SEQ_ID="${SEQUENCES[$CATEGORY]}"
  GT_PLY="${CO3D_ROOT}/${CATEGORY}/${SEQ_ID}/pointcloud.ply"
  if [[ ! -f "${GT_PLY}" ]]; then
    echo "ERROR: Missing GT pointcloud: ${GT_PLY}"
    exit 1
  fi

  for S in ${SIGMAS}; do
    BLUR_DIR="${BLUR_ROOT}/${CATEGORY}/${SEQ_ID}/blur_s${S}"
    if [[ ! -d "${BLUR_DIR}" ]]; then
      echo "ERROR: Missing blur dir: ${BLUR_DIR}"
      exit 1
    fi

    TMP_SEQ="${OUT_ROOT}/tmp_${CATEGORY}_${SEQ_ID}_blur_s${S}"
    mkdir -p "${TMP_SEQ}"
    ln -sfn "${BLUR_DIR}" "${TMP_SEQ}/images"
    ln -sfn "${GT_PLY}" "${TMP_SEQ}/pointcloud.ply"

    OUT_BASELINE="${OUT_ROOT}/${CATEGORY}_${SEQ_ID}_blur_s${S}_baseline"
    OUT_FINETUNED="${OUT_ROOT}/${CATEGORY}_${SEQ_ID}_blur_s${S}_bestval"
    mkdir -p "${OUT_BASELINE}" "${OUT_FINETUNED}"

    echo ""
    echo ">>> ${CATEGORY}/${SEQ_ID} | sigma=${S} | baseline"
    "${VENV_PYTHON}" finetune_blur/deblurdinat/run_inference.py \
      --sequence_dir "${TMP_SEQ}" \
      --dust3r_dir "${DUST3R_DIR}" \
      --dust3r_ckpt "${DUST3R_CKPT}" \
      --output_dir "${OUT_BASELINE}" \
      --image_size "${IMAGE_SIZE}" \
      --n_frames "${N_FRAMES}" \
      --device cuda

    echo ">>> ${CATEGORY}/${SEQ_ID} | sigma=${S} | finetuned(best-val)"
    "${VENV_PYTHON}" finetune_blur/deblurdinat/run_inference.py \
      --sequence_dir "${TMP_SEQ}" \
      --dust3r_dir "${DUST3R_DIR}" \
      --dust3r_ckpt "${DUST3R_CKPT}" \
      --output_dir "${OUT_FINETUNED}" \
      --image_size "${IMAGE_SIZE}" \
      --n_frames "${N_FRAMES}" \
      --device cuda \
      --is_finetuned \
      --finetuned_weights "${BEST_CKPT}" \
      --deblurdinat_repo "${DEBLURDINAT_REPO}"
  done
done

echo ""
echo ">>> Writing per-sequence + averaged Chamfer CSV"
"${VENV_PYTHON}" - <<'PY'
import re
from pathlib import Path
from collections import defaultdict

root = Path("/home/asds/project_Hayk_Minasyan/outputs/dust3r_eval_6seq")
seqs = {
    "teddybear": "101_11758_21048",
    "hydrant": "106_12648_23157",
    "cup": "12_100_593",
    "bottle": "34_1397_4376",
    "toybus": "111_13154_25988",
    "toytrain": "104_12352_22039",
}
sigmas = [5, 10, 20]

def read_cd(path: Path):
    if not path.exists():
        return None
    m = re.search(r"chamfer_distance:\s*([0-9eE.+-]+)", path.read_text())
    return float(m.group(1)) if m else None

rows = []
rows.append("category,seq_id,sigma,baseline_cd,finetuned_cd,delta(baseline-finetuned)")
acc_b = defaultdict(list)
acc_f = defaultdict(list)

for cat, sid in seqs.items():
    for s in sigmas:
        b = read_cd(root / f"{cat}_{sid}_blur_s{s}_baseline" / "metrics.txt")
        f = read_cd(root / f"{cat}_{sid}_blur_s{s}_bestval" / "metrics.txt")
        d = None if (b is None or f is None) else (b - f)
        rows.append(f"{cat},{sid},{s},{b},{f},{d}")
        if b is not None:
            acc_b[s].append(b)
        if f is not None:
            acc_f[s].append(f)

rows.append("")
rows.append("sigma,avg_baseline_cd,avg_finetuned_cd,avg_delta(baseline-finetuned),n_baseline,n_finetuned")
for s in sigmas:
    mb = (sum(acc_b[s]) / len(acc_b[s])) if acc_b[s] else None
    mf = (sum(acc_f[s]) / len(acc_f[s])) if acc_f[s] else None
    md = None if (mb is None or mf is None) else (mb - mf)
    rows.append(f"{s},{mb},{mf},{md},{len(acc_b[s])},{len(acc_f[s])}")

out = root / "blur_6seq_chamfer_compare.csv"
out.write_text("\n".join(rows) + "\n")
print("\n".join(rows))
print(f"\nSaved: {out}")
PY

echo ""
echo "Done: $(date)"
