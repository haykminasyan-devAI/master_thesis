#!/usr/bin/env bash
#SBATCH --job-name=sunet_extreme
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=sunet_extreme_%j.log
#SBATCH --error=sunet_extreme_%j.err

set -euo pipefail

PROJECT_DIR="/home/asds/project_Hayk_Minasyan"
SUNET_DIR="${PROJECT_DIR}/SUNet"
INPUT_DIR="${PROJECT_DIR}/outputs/noise_res/teddybear_camera_noise_test_extreme/noisy"
DENOISED_DIR="${PROJECT_DIR}/outputs/noise_res/teddybear_sunet_extreme/denoised"
COMPARE_DIR="${PROJECT_DIR}/outputs/noise_res/teddybear_sunet_extreme/compare"
ORIG_IMG="${PROJECT_DIR}/outputs/noise_res/teddybear/teddybear_s30_frame000001.jpg"
WEIGHTS="${SUNET_DIR}/pretrained_model/AWGN_denoising_SUNet.pth"

mkdir -p "${DENOISED_DIR}" "${COMPARE_DIR}"

source "/home/asds/miniforge3/etc/profile.d/conda.sh"
conda activate co3d_env

cd "${SUNET_DIR}"
python3 demo_any_resolution.py \
  --input_dir "${INPUT_DIR}" \
  --result_dir "${DENOISED_DIR}" \
  --weights "${WEIGHTS}" \
  --size 256 \
  --stride 128

python3 - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

project = Path("/home/asds/project_Hayk_Minasyan")
orig = Image.open(project / "outputs/noise_res/teddybear/teddybear_s30_frame000001.jpg").convert("RGB")
input_dir = project / "outputs/noise_res/teddybear_camera_noise_test_extreme/noisy"
den_dir = project / "outputs/noise_res/teddybear_sunet_extreme/denoised"
cmp_dir = project / "outputs/noise_res/teddybear_sunet_extreme/compare"
cmp_dir.mkdir(parents=True, exist_ok=True)

for noisy_path in sorted(input_dir.glob("*.*")):
    if noisy_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

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
echo "Denoised: ${DENOISED_DIR}"
echo "Compare : ${COMPARE_DIR}"
