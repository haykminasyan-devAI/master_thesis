#!/usr/bin/env python3
"""
Visualize DUSt3R preprocessing on one image: resize, center-crop, and patch grid.

Shows what patch_size means: the ViT chops the *cropped* image into a grid of
patch_size x patch_size pixel tiles (e.g. 16x16 -> one token per tile).

Example (teddybear frame):
  python scripts/visualize_dust3r_patches.py \\
      --image data/co3d/teddybear/<seq_id>/images/<some>.jpg \\
      --out_dir outputs/dust3r_patch_viz/teddybear_one

Uses only PIL + NumPy (no torch/torchvision) so it runs even if DUSt3R deps are broken.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import PIL.Image
from PIL import ImageDraw
from PIL.ImageOps import exif_transpose


def _resize_pil_image(img: PIL.Image.Image, long_edge_size: int) -> PIL.Image.Image:
    """Same as dust3r.utils.image._resize_pil_image (longest side -> long_edge_size)."""
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def _imagenet_style_norm_rgb01(pil_rgb: PIL.Image.Image) -> PIL.Image.Image:
    """Match DUSt3R ImgNorm then rgb(): ToTensor + Normalize(0.5,0.5), then back to uint8."""
    hwc = np.asarray(pil_rgb.convert("RGB"), dtype=np.float32) / 255.0
    chw = np.transpose(hwc, (2, 0, 1))
    normed = (chw - 0.5) / 0.5
    back = (normed * 0.5) + 0.5
    out = np.transpose(back, (1, 2, 0))
    return PIL.Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8))


def preprocess_like_load_images(
    img: PIL.Image.Image, size: int, patch_size: int, square_ok: bool
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
    """
    Same steps as dust3r.utils.image.load_images for size != 224.
    Returns (pil_after_long_edge_resize, pil_after_center_crop).
    """
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


def draw_patch_grid(pil_rgb: PIL.Image.Image, patch_size: int, color=(255, 0, 0), width=1):
    """Return a copy with vertical/horizontal lines every patch_size pixels."""
    out = pil_rgb.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    for x in range(0, W + 1, patch_size):
        draw.line([(x, 0), (x, H)], fill=color, width=width)
    for y in range(0, H + 1, patch_size):
        draw.line([(0, y), (W, y)], fill=color, width=width)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="Path to one .jpg / .png")
    p.add_argument("--out_dir", required=True, help="Directory for PNG outputs")
    p.add_argument("--size", type=int, default=512, help="Long-edge size (default 512, matches your inference)")
    p.add_argument("--patch_size", type=int, default=16, help="ViT patch size in pixels (default 16)")
    p.add_argument(
        "--square_ok",
        action="store_true",
        help="Pass square_ok=True (disables 4:3-style crop for square resized images)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = exif_transpose(PIL.Image.open(args.image)).convert("RGB")
    W0, H0 = img.size
    resized, cropped = preprocess_like_load_images(
        img, size=args.size, patch_size=args.patch_size, square_ok=args.square_ok
    )
    Wc, Hc = cropped.size

    if Wc % args.patch_size != 0 or Hc % args.patch_size != 0:
        print(
            f"WARNING: cropped size {Wc}x{Hc} is not divisible by patch_size={args.patch_size} "
            "(DUSt3R assumes it is after this crop)."
        )

    nh, nw = Hc // args.patch_size, Wc // args.patch_size
    n_tokens = nh * nw

    stem = os.path.splitext(os.path.basename(args.image))[0]

    path_resized = os.path.join(args.out_dir, f"{stem}_01_resized_long_edge_{args.size}.png")
    path_cropped = os.path.join(args.out_dir, f"{stem}_02_center_cropped.png")
    path_grid = os.path.join(args.out_dir, f"{stem}_03_cropped_patch_grid_{args.patch_size}px.png")
    path_tensor_preview = os.path.join(args.out_dir, f"{stem}_04_after_normalize_like_model.png")

    resized.save(path_resized)
    cropped.save(path_cropped)
    draw_patch_grid(cropped, args.patch_size).save(path_grid)

    # Round-trip same as DUSt3R ImgNorm then visualization (no torch needed)
    preview = _imagenet_style_norm_rgb01(cropped)
    preview.save(path_tensor_preview)

    report = os.path.join(args.out_dir, "summary.txt")
    with open(report, "w") as f:
        f.write(f"source: {args.image}\n")
        f.write(f"original (EXIF-corrected): {W0} x {H0}\n")
        f.write(f"after long-edge resize to {args.size}: {resized.size[0]} x {resized.size[1]}\n")
        f.write(f"after center crop (model spatial input): {Wc} x {Hc}\n")
        f.write(f"patch_size: {args.patch_size} px\n")
        f.write(f"grid: {nw} patches wide x {nh} patches tall = {n_tokens} tokens\n")
        f.write(f"square_ok: {args.square_ok}\n")
        f.write("\noutputs:\n")
        for path in (path_resized, path_cropped, path_grid, path_tensor_preview):
            f.write(f"  {path}\n")

    print(open(report).read())


if __name__ == "__main__":
    main()
