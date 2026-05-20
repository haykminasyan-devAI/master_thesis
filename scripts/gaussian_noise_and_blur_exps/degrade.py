"""
Apply image degradation (Gaussian blur or Gaussian noise) to every frame
in a directory and save the results to a new directory.

Usage – blur:
    python scripts/degrade.py \\
        --images_dir data/co3d/teddybear/101_11758_21048/images \\
        --output_dir outputs/degraded_frames/teddybear/101_11758_21048/blur_sigma3 \\
        --mode blur \\
        --blur_sigma 3.0

Usage – noise:
    python scripts/degrade.py \\
        --images_dir data/co3d/teddybear/101_11758_21048/images \\
        --output_dir outputs/degraded_frames/teddybear/101_11758_21048/noise_std25 \\
        --mode noise \\
        --noise_std 25
"""

import argparse
import os

import numpy as np
from PIL import Image


# ── degradation functions ─────────────────────────────────────────────────────

def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian blur with given sigma (pixels)."""
    from scipy.ndimage import gaussian_filter
    # Apply to each channel independently
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = gaussian_filter(img[:, :, c].astype(np.float32), sigma=sigma)
    return out.clip(0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, std: float, seed: int = 42) -> np.ndarray:
    """Additive zero-mean Gaussian noise with standard deviation `std` (pixel scale 0-255)."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return noisy.clip(0, 255).astype(np.uint8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True,
                        help="Input directory with JPG/PNG frames")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for degraded frames")
    parser.add_argument("--mode", required=True, choices=["blur", "noise"],
                        help="Degradation type: 'blur' or 'noise'")

    # blur options
    parser.add_argument("--blur_sigma", type=float, default=3.0,
                        help="Gaussian blur sigma in pixels (default: 3.0)")

    # noise options
    parser.add_argument("--noise_std", type=float, default=25.0,
                        help="Gaussian noise std dev in [0,255] scale (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise (default: 42)")

    args = parser.parse_args()

    frames = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frames:
        raise SystemExit(f"No images found in {args.images_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "blur":
        label = f"Gaussian blur  σ={args.blur_sigma:.1f}px"
    else:
        label = f"Gaussian noise  σ={args.noise_std:.1f} (0-255)"

    print(f"Found   : {len(frames)} frames")
    print(f"Mode    : {label}")
    print(f"Input   : {args.images_dir}")
    print(f"Output  : {args.output_dir}\n")

    for i, fname in enumerate(frames, 1):
        img = np.array(Image.open(os.path.join(args.images_dir, fname)).convert("RGB"))

        if args.mode == "blur":
            out = apply_gaussian_blur(img, args.blur_sigma)
        else:
            # use a per-frame seed offset so every frame gets independent noise
            out = apply_gaussian_noise(img, args.noise_std, seed=args.seed + i)

        Image.fromarray(out).save(os.path.join(args.output_dir, fname))
        if i % 20 == 0 or i == len(frames):
            print(f"  [{i:>3}/{len(frames)}]  {fname}")

    print(f"\nDone. {len(frames)} degraded frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
