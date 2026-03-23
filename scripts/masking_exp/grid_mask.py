"""
Grid-based patch masking - CroCo style

Divides each image into a regular grid of (patch_size x patch_size) pixel patches,
randomly masks r*N patches by filling them with black pixels, and saves the results.

Usage:
    python scripts/masking_exp/grid_mask.py \
        --images_dir data/co3d/teddybear/101_11758_21048/images \
        --output_dir outputs/grid_masked/teddybear/101_11758_21048/mask_50pct \
        --patch_size 16 \
        --mask_ratio 0.5 \
        --seed 42
"""

import argparse
import os
import random
import numpy as np
from PIL import Image

def grid_mask_image(img: Image.Image, patch_size: int, mask_ratio: float, rng: random.Random) -> Image.Image:
    """
    Divide img into patch_size x patch_size grid, randomly black out mask_ratio fraction.
    Image is cropped to the nearest multiple of patch_size before masking.
    """
    W, H  = img.size

    W_crop = (W // patch_size) * patch_size
    H_crop = (H // patch_size) * patch_size

    img = img.crop((0, 0, W_crop, H_crop))

    arr = np.array(img)

    n_patches_h = H_crop // patch_size
    n_patches_w = W_crop // patch_size
    n_patches = n_patches_h * n_patches_w
    n_mask = int(mask_ratio * n_patches)

    all_indices = list(range(n_patches))
    masked_indices = set(rng.sample(all_indices, n_mask))

    for idx in masked_indices:
        row = idx // n_patches_w
        col = idx  % n_patches_w
        r0, r1 = row * patch_size, (row + 1) * patch_size
        c0, c1 = col * patch_size, (col + 1) * patch_size
        arr[r0:r1, c0:c1] = 0  # black out

    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir",  required=True,  help="Input folder of JPG/PNG frames")
    parser.add_argument("--output_dir",  required=True,  help="Where to save masked frames")
    parser.add_argument("--patch_size",  type=int, default=16, help="Patch size in pixels (default: 16)")
    parser.add_argument("--mask_ratio",  type=float, default=0.5, help="Fraction of patches to mask (default: 0.5)")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    frames = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not frames:
        print(f"No images found in {args.images_dir}")
        return

    print(f"Masking {len(frames)} frames | patch_size={args.patch_size} | mask_ratio={args.mask_ratio}")

    for fname in frames:
        src = os.path.join(args.images_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        img = Image.open(src).convert("RGB")
        masked = grid_mask_image(img, args.patch_size, args.mask_ratio, rng)
        masked.save(dst)

    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()

