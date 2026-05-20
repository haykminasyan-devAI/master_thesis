#!/usr/bin/env bash
#SBATCH --job-name=sunet_xplus
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=sunet_xplus_%j.log
#SBATCH --error=sunet_xplus_%j.err

set -euo pipefail

PROJECT_DIR="/home/asds/project_Hayk_Minasyan"
SUNET_DIR="${PROJECT_DIR}/SUNet"
BASE_IMG="${PROJECT_DIR}/outputs/noise_res/teddybear/teddybear_s30_frame000001.jpg"
ROOT_OUT="${PROJECT_DIR}/outputs/noise_res/teddybear_sunet_extreme_plus"
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
x = np.array(base).astype(np.float32) / 255.0
h, w, _ = x.shape
rng = np.random.default_rng(2026)
out = project / "outputs/noise_res/teddybear_sunet_extreme_plus/noisy"
out.mkdir(parents=True, exist_ok=True)

# Super-high severity settings
shot_strength = 2.2
read_sigma = 0.30
iso_gain = 12.0
iso_read_sigma = 0.18
chan_sigmas = (0.12, 0.32, 0.22)

def to_u8(img01):
    return (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)

def shot_noise(v):
    photons = np.clip(v, 0, 1) * shot_strength
    noisy = rng.poisson(photons).astype(np.float32) / max(shot_strength, 1e-6)
    return np.clip(noisy, 0, 1)

def read_noise(v):
    n = rng.normal(0.0, read_sigma, size=v.shape).astype(np.float32)
    return np.clip(v + n, 0, 1)

def iso_artifacts(v):
    y = v * iso_gain
    y = y + rng.normal(0.0, iso_read_sigma, size=v.shape).astype(np.float32)
    y = np.clip(y, 0, 1)
    # very harsh quantization / banding
    levels = 12
    y = np.round(y * levels) / levels
    # tone-map plus slight gamma distortion
    y = np.power(np.clip(y, 0, 1), 1/3.0)
    return np.clip(y, 0, 1)

def color_channel_noise(v):
    n = np.zeros_like(v, dtype=np.float32)
    for c, s in enumerate(chan_sigmas):
        n[..., c] = rng.normal(0.0, s, size=(h, w)).astype(np.float32)
    return np.clip(v + n, 0, 1)

variants = {
    "shot_noise_photon_xplus": shot_noise(x),
    "read_noise_electronics_xplus": read_noise(x),
    "iso_amplification_artifacts_xplus": iso_artifacts(x),
    "color_channel_dependent_noise_xplus": color_channel_noise(x),
}

for name, arr in variants.items():
    Image.fromarray(to_u8(arr)).save(out / f"{name}.jpg", quality=95)
    print(f"saved {name}")
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
noisy_dir = project / "outputs/noise_res/teddybear_sunet_extreme_plus/noisy"
den_dir = project / "outputs/noise_res/teddybear_sunet_extreme_plus/denoised"
cmp_dir = project / "outputs/noise_res/teddybear_sunet_extreme_plus/compare"
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
