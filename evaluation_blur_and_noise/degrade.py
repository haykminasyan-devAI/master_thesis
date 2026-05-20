"""
Apply image degradation (Gaussian blur or Gaussian noise) to every frame
in a directory and save the results to a new directory.

(Also kept under scripts/gaussian_noise_and_blur_exps/; this copy is for
6-seq eval so preprocess_6seq_blur_noise_eval_asds.sh works without
that untracked path.)
"""

import argparse
import os

import numpy as np
from PIL import Image


def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = gaussian_filter(img[:, :, c].astype(np.float32), sigma=sigma)
    return out.clip(0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, std: float, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return noisy.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["blur", "noise"])
    parser.add_argument("--blur_sigma", type=float, default=3.0)
    parser.add_argument("--noise_std", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frames:
        raise SystemExit(f"No images found in {args.images_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, fname in enumerate(frames, 1):
        img = np.array(Image.open(os.path.join(args.images_dir, fname)).convert("RGB"))
        if args.mode == "blur":
            out = apply_gaussian_blur(img, args.blur_sigma)
        else:
            out = apply_gaussian_noise(img, args.noise_std, seed=args.seed + i)
        Image.fromarray(out).save(os.path.join(args.output_dir, fname))
        if i % 20 == 0 or i == len(frames):
            print(f"  [{i:>3}/{len(frames)}]  {fname}")


if __name__ == "__main__":
    main()
