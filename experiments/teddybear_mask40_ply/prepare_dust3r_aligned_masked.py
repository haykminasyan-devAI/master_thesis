#!/usr/bin/env python3
"""
Build masked frames that match DUSt3R load_images geometry (long-edge resize + center crop,
patch-aligned), then randomly black out a fraction of patch_size x patch_size tiles.

Saves images with the SAME basenames as in images/ so run_dust3r_inference.py can use
--masked_dir with --n_masked == n_frames.

Also writes patch masks to --mask_npy_dir as <fname>.patch_mask_hw.npy (bool [nh,nw], True=masked)
for attention masking (see scripts/run_dust3r_inference_attn_mask.py).

Usage:
  python experiments/teddybear_mask40_ply/prepare_dust3r_aligned_masked.py \\
    --images_dir data/co3d/teddybear/101_11758_21048/images \\
    --out_dir    experiments/teddybear_mask40_ply/masked_images \\
    --n_frames 15 --mask_ratio 0.4 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose


def _resize_pil_image(img: PIL.Image.Image, long_edge_size: int) -> PIL.Image.Image:
    S = max(img.size)
    interp = PIL.Image.LANCZOS if S > long_edge_size else PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def preprocess_like_load_images(
    img: PIL.Image.Image, size: int, patch_size: int, square_ok: bool
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
    W1, H1 = img.size
    if size == 224:
        resized = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        resized = _resize_pil_image(img, size)
    W, H = resized.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        cropped = resized.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not square_ok and W == H:
            halfh = 3 * halfw / 4
        cropped = resized.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return resized, cropped


def select_frames(images_dir: str, n_frames: int) -> list[str]:
    all_frames = sorted(
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(all_frames) <= n_frames:
        return all_frames
    indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
    return [all_frames[i] for i in indices]


def apply_random_patch_mask(
    arr_hwc: np.ndarray, patch_size: int, mask_ratio: float, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Black out mask_ratio of non-overlapping patch_size tiles. arr uint8 HxWx3.

    Returns (masked_rgb, patch_mask_hw) where patch_mask_hw[row,col] is True iff that patch is masked.
    Row/col follow tensor H,W patch grid (same order as PatchEmbedDust3R token raster).
    """
    H, W = arr_hwc.shape[:2]
    assert H % patch_size == 0 and W % patch_size == 0, f"{H}x{W} not divisible by {patch_size}"
    nh, nw = H // patch_size, W // patch_size
    n_patches = nh * nw
    n_mask = int(mask_ratio * n_patches)
    out = arr_hwc.copy()
    patch_mask_hw = np.zeros((nh, nw), dtype=bool)
    all_i = list(range(n_patches))
    masked = set(rng.sample(all_i, n_mask))
    for idx in masked:
        row, col = idx // nw, idx % nw
        patch_mask_hw[row, col] = True
        r0, r1 = row * patch_size, (row + 1) * patch_size
        c0, c1 = col * patch_size, (col + 1) * patch_size
        out[r0:r1, c0:c1] = 0
    return out, patch_mask_hw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_frames", type=int, default=15)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--long_edge", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--square_ok", action="store_true")
    p.add_argument(
        "--mask_npy_dir",
        default=None,
        help="Directory for .patch_mask_hw.npy per frame (default: <out_dir>_patch_masks)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_npy_dir = args.mask_npy_dir or (args.out_dir.rstrip("/") + "_patch_masks")
    os.makedirs(mask_npy_dir, exist_ok=True)
    frames = select_frames(args.images_dir, args.n_frames)
    if not frames:
        raise SystemExit(f"No images in {args.images_dir}")

    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), "prepare_meta.txt")
    lines = [
        f"images_dir: {args.images_dir}",
        f"out_dir: {args.out_dir}",
        f"n_frames: {args.n_frames}",
        f"mask_ratio: {args.mask_ratio}",
        f"seed: {args.seed}",
        f"long_edge: {args.long_edge}",
        f"patch_size: {args.patch_size}",
        f"square_ok: {args.square_ok}",
        f"mask_npy_dir: {mask_npy_dir}",
        "frames:",
    ]

    for i, fname in enumerate(frames):
        src = os.path.join(args.images_dir, fname)
        img = exif_transpose(PIL.Image.open(src)).convert("RGB")
        _resized, cropped = preprocess_like_load_images(
            img, size=args.long_edge, patch_size=args.patch_size, square_ok=args.square_ok
        )
        arr = np.array(cropped)
        rng = random.Random(args.seed + i)
        masked_arr, patch_mask_hw = apply_random_patch_mask(arr, args.patch_size, args.mask_ratio, rng)
        npy_name = f"{fname}.patch_mask_hw.npy"
        np.save(os.path.join(mask_npy_dir, npy_name), patch_mask_hw)
        out_img = PIL.Image.fromarray(masked_arr)
        dst = os.path.join(args.out_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in (".jpg", ".jpeg"):
            out_img.save(dst, quality=95)
        else:
            out_img.save(dst)

        H, W = masked_arr.shape[:2]
        nph, npw = H // args.patch_size, W // args.patch_size
        lines.append(f"  {fname}  cropped={W}x{H}  grid={npw}x{nph}  patches={nph*npw}")

    with open(meta_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(frames)} masked frames → {args.out_dir}")
    print(f"Patch mask npy → {mask_npy_dir}")
    print(f"Meta → {meta_path}")


if __name__ == "__main__":
    main()
