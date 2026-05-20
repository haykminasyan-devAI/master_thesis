#!/usr/bin/env bash
#SBATCH --job-name=sunet_gauss
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=sunet_gauss_%j.log
#SBATCH --error=sunet_gauss_%j.err

set -euo pipefail

PROJECT_DIR="/home/asds/project_Hayk_Minasyan"
SUNET_DIR="${PROJECT_DIR}/SUNet"
BASE_IMG="${PROJECT_DIR}/outputs/noise_res/teddybear/teddybear_s30_frame000001.jpg"
ROOT_OUT="${PROJECT_DIR}/outputs/noise_res/teddybear_sunet_gaussian_70_80_120_140"
NOISY_DIR="${ROOT_OUT}/noisy"
DENOISED_DIR="${ROOT_OUT}/denoised"
COMPARE_DIR="${ROOT_OUT}/compare"
WEIGHTS="${SUNET_DIR}/pretrained_model/AWGN_denoising_SUNet.pth"

mkdir -p "${NOISY_DIR}" "${DENOISED_DIR}" "${COMPARE_DIR}"

source "/home/asds/miniforge3/etc/profile.d/conda.sh"
conda activate co3d_env

python3 - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

project = Path("/home/asds/project_Hayk_Minasyan")
base = Image.open(project / "outputs/noise_res/teddybear/teddybear_s30_frame000001.jpg").convert("RGB")
arr = np.array(base).astype(np.float32)
out = project / "outputs/noise_res/teddybear_sunet_gaussian_70_80_120_140/noisy"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1234)

for sigma in [70, 80, 120, 140]:
    noisy = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32), 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(out / f"teddybear_gaussian_sigma{sigma}.jpg", quality=95)
    print(f"saved sigma={sigma}")
PY

cd "${SUNET_DIR}"
python3 demo_any_resolution.py \
  --input_dir "${NOISY_DIR}" \
  --result_dir "${DENOISED_DIR}" \
  --weights "${WEIGHTS}" \
  --size 256 \
  --stride 128

python3 - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

project = Path("/home/asds/project_Hayk_Minasyan")
orig = Image.open(project / "outputs/noise_res/teddybear/teddybear_s30_frame000001.jpg").convert("RGB")
noisy_dir = project / "outputs/noise_res/teddybear_sunet_gaussian_70_80_120_140/noisy"
den_dir = project / "outputs/noise_res/teddybear_sunet_gaussian_70_80_120_140/denoised"
cmp_dir = project / "outputs/noise_res/teddybear_sunet_gaussian_70_80_120_140/compare"
cmp_dir.mkdir(parents=True, exist_ok=True)

for noisy_path in sorted(noisy_dir.glob("*.jpg")):
    stem = noisy_path.stem
    den_path = den_dir / f"{stem}.png"
    if not den_path.exists():
        continue
    noisy = Image.open(noisy_path).convert("RGB")
    den = Image.open(den_path).convert("RGB")

    panel = Image.new("RGB", (orig.width * 3, orig.height + 34), (18, 18, 18))
    panel.paste(orig, (0, 34))
    panel.paste(noisy, (orig.width, 34))
    panel.paste(den, (orig.width * 2, 34))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), "original", fill=(255, 255, 255))
    draw.text((orig.width + 10, 10), f"noisy: {stem}", fill=(255, 255, 255))
    draw.text((orig.width * 2 + 10, 10), "sunet denoised", fill=(255, 255, 255))
    out = cmp_dir / f"{stem}_sunet_compare.jpg"
    panel.save(out, quality=95)
    print(f"saved {out}")
PY

echo "DONE"
echo "Noisy   : ${NOISY_DIR}"
echo "Denoised: ${DENOISED_DIR}"
echo "Compare : ${COMPARE_DIR}"
