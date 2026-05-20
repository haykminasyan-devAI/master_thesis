#!/usr/bin/env bash
#SBATCH --job-name=teddy_blur_cmp
#SBATCH --chdir=/home/asds/project_Hayk_Minasyan
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/asds/project_Hayk_Minasyan/logs/teddy_blur_cmp_%j.log
#SBATCH --error=/home/asds/project_Hayk_Minasyan/logs/teddy_blur_cmp_%j.err

set -euo pipefail

: "${PROJECT_DIR:=/home/asds/project_Hayk_Minasyan}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

source /home/asds/miniforge3/etc/profile.d/conda.sh
conda activate co3d_env
VENV_PYTHON="$(command -v python3)"

# Sequence/config (override by exporting before sbatch if needed)
: "${CATEGORY:=teddybear}"
: "${SEQ_ID:=101_11758_21048}"
: "${SIGMAS:=5 10 20}"
: "${N_FRAMES:=20}"
: "${IMAGE_SIZE:=224}"

: "${DUST3R_DIR:=${PROJECT_DIR}/dust3r}"
: "${DUST3R_CKPT:=${PROJECT_DIR}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}"
: "${DEBLURDINAT_REPO:=${PROJECT_DIR}/DeblurDiNAT}"
: "${BEST_CKPT:=${PROJECT_DIR}/finetune_blur_runs/deblurdinat_dust3r_asds_224/joint_sigmas_5_10_20/checkpoint-best-val.pth}"

: "${BLUR_BASE:=${PROJECT_DIR}/outputs/degraded_frames/${CATEGORY}/${SEQ_ID}}"
: "${GT_PLY:=${PROJECT_DIR}/data/co3d/${CATEGORY}/${SEQ_ID}/pointcloud.ply}"
: "${OUT_ROOT:=${PROJECT_DIR}/outputs/dust3r_eval}"

if [[ ! -f "${DUST3R_CKPT}" ]]; then
  echo "ERROR: Missing DUSt3R checkpoint: ${DUST3R_CKPT}"
  exit 1
fi
if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "ERROR: Missing finetuned checkpoint: ${BEST_CKPT}"
  exit 1
fi
if [[ ! -f "${GT_PLY}" ]]; then
  echo "ERROR: Missing GT pointcloud: ${GT_PLY}"
  exit 1
fi

echo "============================================================"
echo "Blur comparison job | $(date) | $(hostname)"
echo "Category/Seq : ${CATEGORY}/${SEQ_ID}"
echo "Sigmas       : ${SIGMAS}"
echo "DUSt3R ckpt  : ${DUST3R_CKPT}"
echo "Best ckpt    : ${BEST_CKPT}"
echo "Output root  : ${OUT_ROOT}"
echo "============================================================"

for S in ${SIGMAS}; do
  BLUR_DIR="${BLUR_BASE}/blur_s${S}"
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
  echo ">>> Sigma ${S}: baseline DUSt3R"
  "${VENV_PYTHON}" finetune_blur/deblurdinat/run_inference.py \
    --sequence_dir "${TMP_SEQ}" \
    --dust3r_dir "${DUST3R_DIR}" \
    --dust3r_ckpt "${DUST3R_CKPT}" \
    --output_dir "${OUT_BASELINE}" \
    --image_size "${IMAGE_SIZE}" \
    --n_frames "${N_FRAMES}" \
    --device cuda

  echo ">>> Sigma ${S}: finetuned (best-val)"
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

echo ""
echo ">>> Chamfer comparison (lower is better)"
"${VENV_PYTHON}" - <<'PY'
import re
from pathlib import Path

root = Path("/home/asds/project_Hayk_Minasyan/outputs/dust3r_eval")
cat = "teddybear"
seq = "101_11758_21048"
sigmas = [5, 10, 20]

def read_cd(path: Path):
    if not path.exists():
        return None
    m = re.search(r"chamfer_distance:\s*([0-9eE.+-]+)", path.read_text())
    return float(m.group(1)) if m else None

lines = []
lines.append("sigma,baseline_cd,finetuned_cd,delta(baseline-finetuned)")
for s in sigmas:
    b = read_cd(root / f"{cat}_{seq}_blur_s{s}_baseline" / "metrics.txt")
    f = read_cd(root / f"{cat}_{seq}_blur_s{s}_bestval" / "metrics.txt")
    d = None if (b is None or f is None) else (b - f)
    lines.append(f"{s},{b},{f},{d}")

out = root / f"{cat}_{seq}_blur_chamfer_compare.csv"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved: {out}")
PY

echo ""
echo "Done: $(date)"
