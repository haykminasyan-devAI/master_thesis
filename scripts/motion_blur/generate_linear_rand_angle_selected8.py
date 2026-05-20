#!/usr/bin/env python3
"""
Build motion-blur frames from raw CO3D using *_selected_8.json per category.

For each frame, applies a normalized line kernel of fixed odd length with angle
drawn uniformly in [0, 360). RNG is seeded so reruns are deterministic per
(category, sequence_id). Output layout matches Co3dMotion:

  motion_root/<category>/<seq_id>/<motion_tag>/frameXXXXXX.jpg

Kernel helpers mirror scripts/motion_blur/generate_motion_blur_examples.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
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


def list_frames(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in images_dir.iterdir() if p.suffix in exts]
    paths.sort(key=lambda p: (int(m.group(1)) if (m := re.search(r"(\d+)", p.stem)) else p.stem))
    if not paths:
        raise FileNotFoundError(f"No images in {images_dir}")
    return paths


def rng_for_sequence(base_seed: int, category: str, seq_id: str) -> np.random.Generator:
    h = zlib.adler32(f"{category}/{seq_id}".encode("utf-8"))
    return np.random.default_rng((base_seed + h) % (2**32))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_co3d_root", type=Path, required=True, help="e.g. /mnt/weka/hminasyan/data/co3d (contains *_selected_8.json)")
    p.add_argument("--motion_root", type=Path, required=True, help="Output root, e.g. .../degraded_frames_motion_10cat")
    p.add_argument(
        "--motion_tag",
        default="linear_rand_angle_0_360_L31_seed123",
        help="Subfolder name under each sequence (must match training MOTION_TAG)",
    )
    p.add_argument("--kernel_len", type=int, default=31, help="Odd motion kernel length (e.g. 31)")
    p.add_argument("--seed", type=int, default=123, help="Base RNG seed (tag name suffix)")
    p.add_argument(
        "--categories",
        nargs="+",
        default=[
            "apple",
            "banana",
            "baseballbat",
            "baseballglove",
            "bicycle",
            "bowl",
            "broccoli",
            "cake",
            "car",
            "carrot",
        ],
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    L = int(args.kernel_len)
    if L < 3 or L % 2 == 0:
        print("ERROR: --kernel_len must be odd and >= 3", file=sys.stderr)
        sys.exit(1)

    raw_root = args.raw_co3d_root.resolve()
    motion_root = args.motion_root.resolve()
    tag = str(args.motion_tag)

    total_written = 0
    total_skipped = 0

    for cat in args.categories:
        sel_path = raw_root / f"{cat}_selected_8.json"
        if not sel_path.is_file():
            print(f"SKIP (missing json): {sel_path}", file=sys.stderr)
            continue
        selected = json.loads(sel_path.read_text())
        if not isinstance(selected, list) or len(selected) != 8:
            print(f"SKIP (need 8 seq ids): {sel_path}", file=sys.stderr)
            continue

        for sid in selected:
            images_dir = raw_root / cat / sid / "images"
            if not images_dir.is_dir():
                print(f"SKIP (no images dir): {images_dir}", file=sys.stderr)
                continue
            out_dir = motion_root / cat / sid / tag
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)

            paths = list_frames(images_dir)
            rng = rng_for_sequence(args.seed, cat, sid)
            n_paths = len(paths)
            print(f"  {cat}/{sid}: processing {n_paths} frames -> {out_dir}", flush=True)

            for fi, src in enumerate(paths):
                if (fi + 1) == 1 or (fi + 1) % 100 == 0 or (fi + 1) == n_paths:
                    print(f"    {cat}/{sid}: {fi + 1}/{n_paths}", flush=True)
                dst = out_dir / src.name
                if dst.exists() and not args.force:
                    total_skipped += 1
                    continue
                angle = float(rng.uniform(0.0, 360.0))
                k = motion_kernel(L, angle)
                img = np.array(Image.open(src).convert("RGB"), dtype=np.uint8)
                blurred = apply_blur_rgb(img, k)
                if not args.dry_run:
                    Image.fromarray(blurred).save(dst, quality=95)
                total_written += 1

            print(f"[ok] {cat}/{sid}: frames={n_paths} -> {out_dir}", flush=True)

    print(
        f"Done. wrote={total_written} skipped_existing={total_skipped} dry_run={args.dry_run} tag={tag}",
        flush=True,
    )


if __name__ == "__main__":
    main()
