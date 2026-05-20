import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def apply_gaussian_noise_debug(img: np.ndarray, std: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy_uint8 = noisy.clip(0, 255).astype(np.uint8)
    return noise, noisy, noisy_uint8


def stats(name: str, arr: np.ndarray):
    print(
        f"{name:>12}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="/home/asds/project_Hayk_Minasyan/outputs/noisy_frames_10cat/baseballglove/117_13765_29509/noise_s70/frame000001.jpg",
    )
    parser.add_argument("--std", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="/home/asds/project_Hayk_Minasyan/outputs/noisy_frames_10cat/baseballglove/117_13765_29509/noise_s70/frame000001_debug_noisy.jpg",
    )
    args = parser.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    noise, noisy, noisy_uint8 = apply_gaussian_noise_debug(img, std=args.std, seed=args.seed)

    stats("img", img)
    stats("noise", noise)
    stats("noisy", noisy)
    stats("noisy_uint8", noisy_uint8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy_uint8).save(args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
