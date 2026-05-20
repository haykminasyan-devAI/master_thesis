#!/usr/bin/env python3
"""
Generate motion-blur examples from a single image.

Creates:
- linear horizontal/vertical blur examples
- angle-based motion blur examples (e.g., 15, 30, 45, 60 deg)
- camera-shake style blur (trajectory-based averaging)
- spatially varying patchwise motion blur
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import cv2


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Create a normalized line kernel with given length and angle."""
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1

    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    t_vals = np.linspace(-(length // 2), length // 2, length)
    for t in t_vals:
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0

    s = k.sum()
    if s <= 0:
        k[c, :] = 1.0
        s = k.sum()
    return k / s


def apply_blur_rgb(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = np.empty_like(img)
    for ch in range(img.shape[2]):
        out[:, :, ch] = convolve(img[:, :, ch].astype(np.float32), kernel, mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_trajectory(length: int, max_shift: float, rng: np.random.Generator) -> np.ndarray:
    """
    Random walk trajectory for camera-shake simulation.
    Returns array of shape [length, 2] with (dx, dy).
    """
    length = max(2, int(length))
    max_shift = float(max(0.5, max_shift))
    x = 0.0
    y = 0.0
    traj = []
    for _ in range(length):
        x = float(np.clip(x + rng.normal(0.0, 0.5), -max_shift, max_shift))
        y = float(np.clip(y + rng.normal(0.0, 0.5), -max_shift, max_shift))
        traj.append((x, y))
    return np.asarray(traj, dtype=np.float32)


def apply_camera_shake_blur(img: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """Apply camera-shake blur by averaging translated copies."""
    h, w = img.shape[:2]
    accum = np.zeros_like(img, dtype=np.float32)
    for dx, dy in trajectory:
        m = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
        shifted = cv2.warpAffine(
            img,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        accum += shifted.astype(np.float32)
    accum /= max(1, len(trajectory))
    return np.clip(accum, 0, 255).astype(np.uint8)


def spatial_blur_patchwise(
    img: np.ndarray,
    grid_size: int,
    min_kernel: int,
    max_kernel: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply random directional motion blur per grid patch.
    """
    h, w = img.shape[:2]
    grid_size = max(1, int(grid_size))
    min_kernel = max(3, int(min_kernel))
    max_kernel = max(min_kernel, int(max_kernel))
    if min_kernel % 2 == 0:
        min_kernel += 1
    if max_kernel % 2 == 0:
        max_kernel -= 1
        if max_kernel < min_kernel:
            max_kernel = min_kernel

    out = img.copy()
    ys = np.linspace(0, h, grid_size + 1, dtype=int)
    xs = np.linspace(0, w, grid_size + 1, dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            angle = float(rng.uniform(0.0, 180.0))
            k = int(rng.integers(min_kernel, max_kernel + 1))
            if k % 2 == 0:
                k += 1
            kernel = motion_kernel(k, angle)
            out[y0:y1, x0:x1] = apply_blur_rgb(patch, kernel)
    return out


def save(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--length", type=int, default=31, help="Motion kernel length")
    ap.add_argument("--angles", type=str, default="15,30,45,60,75")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shake_length", type=int, default=20, help="Trajectory points for camera shake")
    ap.add_argument("--shake_max_shift", type=float, default=5.0, help="Max pixel shift for camera shake")
    ap.add_argument("--grid_size", type=int, default=4, help="Patch grid size for spatial blur")
    ap.add_argument("--patch_min_kernel", type=int, default=5)
    ap.add_argument("--patch_max_kernel", type=int, default=19)
    args = ap.parse_args()

    image_path = Path(args.image_path)
    out_dir = Path(args.output_dir)
    angles = [float(x.strip()) for x in args.angles.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    save(img, out_dir / "original.png")

    # Linear examples
    k_h = motion_kernel(args.length, 0.0)
    k_v = motion_kernel(args.length, 90.0)
    save(apply_blur_rgb(img, k_h), out_dir / f"linear_horizontal_L{args.length}.png")
    save(apply_blur_rgb(img, k_v), out_dir / f"linear_vertical_L{args.length}.png")

    # Angle examples
    for a in angles:
        k = motion_kernel(args.length, a)
        save(apply_blur_rgb(img, k), out_dir / f"angle_{int(a)}deg_L{args.length}.png")

    # Camera shake example
    traj = generate_trajectory(args.shake_length, args.shake_max_shift, rng)
    cam_blur = apply_camera_shake_blur(img, traj)
    save(
        cam_blur,
        out_dir / f"camera_shake_len{args.shake_length}_shift{args.shake_max_shift:g}_seed{args.seed}.png",
    )

    # Spatially varying patchwise blur example
    spatial = spatial_blur_patchwise(
        img=img,
        grid_size=args.grid_size,
        min_kernel=args.patch_min_kernel,
        max_kernel=args.patch_max_kernel,
        rng=rng,
    )
    save(
        spatial,
        out_dir
        / f"spatial_patchwise_g{args.grid_size}_k{args.patch_min_kernel}_{args.patch_max_kernel}_seed{args.seed}.png",
    )

    print(f"Saved examples to: {out_dir}")


if __name__ == "__main__":
    main()

