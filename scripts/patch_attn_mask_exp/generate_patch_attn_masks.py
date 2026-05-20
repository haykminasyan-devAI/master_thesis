#!/usr/bin/env python3
"""
Build per-frame patch attention masks aligned with DUSt3R preprocessing.

DUSt3R: long-edge resize (default 512), center crop with sizes snapped to multiples
of patch_size (16), then ViT tokens = one per patch cell.

Writes one file per image:
    <basename>.patch_mask_hw.npy   — bool array (nh, nw), True = masked patch
Those keys are blocked in encoder self-attention and decoder cross-attention
when passed via run_dust3r_inference.py --attn_mask_npy_dir.

Usage:
    python scripts/patch_attn_mask_exp/generate_patch_attn_masks.py \\
        --images_dir data/co3d/teddybear/101_11758_21048/images \\
        --output_dir outputs/patch_attn_masks/teddybear/101_11758_21048/mask_25pct \\
        --mask_ratio 0.25 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import os
import zlib

import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose


def _resize_pil_image(img: PIL.Image.Image, long_edge_size: int) -> PIL.Image.Image:
    s = max(img.size)
    interp = PIL.Image.LANCZOS if s > long_edge_size else PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / s)) for x in img.size)
    return img.resize(new_size, interp)


def dust3r_crop_size(
    img: PIL.Image.Image, long_edge: int, patch_size: int, square_ok: bool
) -> tuple[int, int]:
    """Return (H, W) of the PIL image after the same crop as dust3r.utils.image.load_images."""
    w1, h1 = img.size
    if long_edge == 224:
        resized = _resize_pil_image(img, round(long_edge * max(w1 / h1, h1 / w1)))
    else:
        resized = _resize_pil_image(img, long_edge)

    w, h = resized.size
    cx, cy = w // 2, h // 2
    if long_edge == 224:
        half = min(cx, cy)
        cropped = resized.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not square_ok and w == h:
            halfh = 3 * halfw / 4
        cropped = resized.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    wc, hc = cropped.size
    assert hc % patch_size == 0 and wc % patch_size == 0, (wc, hc, patch_size)
    return hc, wc


def random_mask_hw(nh: int, nw: int, mask_ratio: float, rng: np.random.Generator) -> np.ndarray:
    n = nh * nw
    n_mask = int(round(mask_ratio * n))
    n_mask = max(0, min(n, n_mask))
    flat = np.zeros(n, dtype=bool)
    if n_mask > 0:
        flat[rng.choice(n, size=n_mask, replace=False)] = True
    return flat.reshape(nh, nw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images_dir", required=True, help="Folder of JPG/PNG frames (same as CO3D images/)")
    p.add_argument("--output_dir", required=True, help="Directory for *.patch_mask_hw.npy")
    p.add_argument("--long_edge", type=int, default=512, help="DUSt3R long-edge size (default 512)")
    p.add_argument("--patch_size", type=int, default=16, help="ViT patch size (default 16)")
    p.add_argument("--mask_ratio", type=float, required=True, help="Fraction of patches to mask [0,1]")
    p.add_argument("--seed", type=int, default=42, help="Base seed (per-frame seed mixes in filename)")
    p.add_argument(
        "--square_ok",
        action="store_true",
        help="Match load_images(..., square_ok=True) (no 4:3 crop tweak for square inputs)",
    )
    args = p.parse_args()

    if not 0.0 <= args.mask_ratio <= 1.0:
        raise SystemExit("--mask_ratio must be in [0, 1]")

    os.makedirs(args.output_dir, exist_ok=True)

    frames = sorted(
        f
        for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frames:
        raise SystemExit(f"No images in {args.images_dir}")

    print(
        f"Writing {len(frames)} masks → {args.output_dir}\n"
        f"  long_edge={args.long_edge} patch_size={args.patch_size} "
        f"mask_ratio={args.mask_ratio} square_ok={args.square_ok}"
    )

    for fname in frames:
        path = os.path.join(args.images_dir, fname)
        img = exif_transpose(PIL.Image.open(path)).convert("RGB")
        h, w = dust3r_crop_size(img, args.long_edge, args.patch_size, args.square_ok)
        nh, nw = h // args.patch_size, w // args.patch_size
        subseed = (args.seed + zlib.adler32(fname.encode("utf-8"))) & 0xFFFFFFFF
        rng = np.random.default_rng(subseed)
        m = random_mask_hw(nh, nw, args.mask_ratio, rng)
        out_path = os.path.join(args.output_dir, f"{fname}.patch_mask_hw.npy")
        np.save(out_path, m)

    print("Done.")


if __name__ == "__main__":
    main()
