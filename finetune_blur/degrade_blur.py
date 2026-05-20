#!/usr/bin/env python3
"""Gaussian blur on all frames in a directory (standalone; no scripts/ dependency)."""

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--blur_sigma', type=float, required=True)
    args = p.parse_args()

    if not os.path.isdir(args.images_dir):
        print(f'[WARN] images_dir does not exist, skipping: {args.images_dir}')
        return

    frames = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not frames:
        raise SystemExit(f'No images found in {args.images_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Blur σ={args.blur_sigma}: {len(frames)} frames -> {args.output_dir}')

    for i, fname in enumerate(frames, 1):
        path = os.path.join(args.images_dir, fname)
        img = np.array(Image.open(path).convert('RGB'))
        out = apply_gaussian_blur(img, args.blur_sigma)
        Image.fromarray(out).save(os.path.join(args.output_dir, fname))
        if i % 50 == 0 or i == len(frames):
            print(f'  [{i}/{len(frames)}] {fname}')

    print('Done.')


if __name__ == '__main__':
    main()
