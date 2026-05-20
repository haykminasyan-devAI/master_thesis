import os
import argparse
import numpy as np
from PIL import Image

def apply_random_mask_multi_no_overlap(img: np.ndarray, mask_ratio: float, num_patches: int = 3) -> np.ndarray:
    """Place exactly num_patches non-overlapping black rectangles covering mask_ratio of the image.

    Uses a grid-based strategy: divide the image into a grid of cells (one per patch),
    then place each patch inside its assigned cell with random jitter. This guarantees
    exactly num_patches patches are placed with zero overlap, regardless of mask_ratio
    or num_patches size.
    """
    H, W = img.shape[:2]

    area_per_patch = (H * W * mask_ratio) / num_patches
    side = max(1, int(area_per_patch ** 0.5))

    # Build a grid with at least num_patches cells.
    # Start from a square grid and expand if needed.
    cols = max(1, int(np.ceil(np.sqrt(num_patches))))
    rows = int(np.ceil(num_patches / cols))

    # Shrink patch side if it doesn't fit inside a single cell
    cell_w = W // cols
    cell_h = H // rows
    side = min(side, cell_w, cell_h)
    side = max(side, 1)

    # All available cell origins, shuffled
    cells = [(c * cell_w, r * cell_h) for r in range(rows) for c in range(cols)]
    np.random.shuffle(cells)
    selected = cells[:num_patches]

    img = img.copy()
    for (cx, cy) in selected:
        max_jitter_x = max(0, cell_w - side)
        max_jitter_y = max(0, cell_h - side)
        jitter_x = np.random.randint(0, max_jitter_x + 1) if max_jitter_x > 0 else 0
        jitter_y = np.random.randint(0, max_jitter_y + 1) if max_jitter_y > 0 else 0
        x = min(cx + jitter_x, W - side)
        y = min(cy + jitter_y, H - side)
        img[y:y + side, x:x + side] = 0

    return img

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply random non-overlapping masks to all frames in a directory.")
    parser.add_argument("--images_dir",  required=True,  help="Path to folder with input frames (jpg/png)")
    parser.add_argument("--output_dir",  required=True,  help="Path to save masked frames")
    parser.add_argument("--mask_ratio",  type=float, required=True,
                        help="Fraction of pixels to black out, e.g. 0.25 for 25%%")
    parser.add_argument("--num_patches", type=int, default=3,
                        help="Number of non-overlapping patches (default: 3)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # collect all image files sorted
    frames = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(frames) == 0:
        print(f"No images found in {args.images_dir}")
        exit(1)

    print(f"Found {len(frames)} frames in {args.images_dir}")
    print(f"Mask ratio: {args.mask_ratio*100:.0f}%  |  Patches: {args.num_patches}  |  Seed: {args.seed}")
    print(f"Saving to: {args.output_dir}\n")

    for fname in frames:
        img = np.array(Image.open(os.path.join(args.images_dir, fname)).convert("RGB"))
        masked = apply_random_mask_multi_no_overlap(img, args.mask_ratio, args.num_patches)
        Image.fromarray(masked).save(os.path.join(args.output_dir, fname))
        print(f"  masked: {fname}")

    print(f"\nDone. {len(frames)} masked frames saved to {args.output_dir}")


    



