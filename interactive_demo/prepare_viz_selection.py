#!/usr/bin/env python3
"""Prepare 20 selected frames + motion/defocus variants for demo uploads."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


def motion_kernel_25() -> np.ndarray:
    k = np.zeros((25, 25), dtype=np.float32)
    k[12, :] = 1.0
    k /= k.sum()
    return k


def defocus_kernel_r7() -> np.ndarray:
    r = 7
    s = 2 * r + 1
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x * x + y * y) <= (r * r)
    k = np.zeros((s, s), dtype=np.float32)
    k[mask] = 1.0
    k /= k.sum()
    return k


def conv2d_rgb_reflect(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img.astype(np.float32), ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            patch = pad[y : y + kh, x : x + kw, :]
            out[y, x] = (patch * k[:, :, None]).sum(axis=(0, 1))
    return np.clip(out, 0, 255).astype(np.uint8)


def pick_evenly_spaced(files: list[Path], n: int) -> list[Path]:
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx]


def main():
    ap = argparse.ArgumentParser("Prepare viz selection with blur variants")
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--n_frames", type=int, default=20)
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_root = Path(args.out_root)
    clean_dir = out_root / "clean"
    motion_dir = out_root / "motion_blur"
    defocus_dir = out_root / "defocus_blur"
    for d in (clean_dir, motion_dir, defocus_dir):
        d.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not files:
        raise RuntimeError(f"No images found in {src_dir}")
    picks = pick_evenly_spaced(files, args.n_frames)

    km = motion_kernel_25()
    kd = defocus_kernel_r7()

    for i, p in enumerate(picks):
        rgb = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        motion = conv2d_rgb_reflect(rgb, km)
        defocus = conv2d_rgb_reflect(rgb, kd)
        name = f"frame_{i:03d}{p.suffix.lower() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} else '.jpg'}"
        Image.fromarray(rgb, mode="RGB").save(clean_dir / name)
        Image.fromarray(motion, mode="RGB").save(motion_dir / name)
        Image.fromarray(defocus, mode="RGB").save(defocus_dir / name)

    print(f"Saved {len(picks)} frames to:")
    print(f"  clean   : {clean_dir}")
    print(f"  motion  : {motion_dir}")
    print(f"  defocus : {defocus_dir}")


if __name__ == "__main__":
    main()

