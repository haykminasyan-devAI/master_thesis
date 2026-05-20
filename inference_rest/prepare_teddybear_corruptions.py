#!/usr/bin/env python3
"""Collect teddybear corrupted frames into inference_rest/corrupted/teddybear."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

CORRUPTION_KEYWORDS = {
    "noise": ("noise", "gaussian_noise", "gauss_noise"),
    "blur": ("blur",),
    "motion_blur": ("motion_blur", "motionblur"),
    "raining": ("rain", "raining", "rainy"),
    "defocus_blur": ("defocus", "defocus_blur"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source_root", required=True, help="Root path containing corrupted frames.")
    p.add_argument(
        "--target_root",
        default="inference_rest/corrupted/teddybear",
        help="Destination root organized by corruption type.",
    )
    p.add_argument(
        "--limit_per_corruption",
        type=int,
        default=300,
        help="Max frames to collect per corruption.",
    )
    p.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="copy (safe) or symlink (fast, no duplication).",
    )
    return p.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def find_matches(source_root: Path, keywords: tuple[str, ...], limit: int) -> list[Path]:
    out: list[Path] = []
    for p in source_root.rglob("*"):
        if not p.is_file() or not is_image(p):
            continue
        low = str(p).lower()
        if "teddybear" not in low:
            continue
        if not any(k in low for k in keywords):
            continue
        out.append(p)
    out.sort()
    return out[:limit]


def unique_name(src: Path) -> str:
    digest = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:8]
    return f"{src.stem}_{digest}{src.suffix.lower()}"


def transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    print(f"source_root: {source_root}")
    print(f"target_root: {target_root}")
    print(f"mode: {args.mode}, limit_per_corruption: {args.limit_per_corruption}")

    for corr, keywords in CORRUPTION_KEYWORDS.items():
        matches = find_matches(source_root, keywords, args.limit_per_corruption)
        corr_dir = target_root / corr
        corr_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in matches:
            dst = corr_dir / unique_name(src)
            transfer(src, dst, args.mode)
            copied += 1
        print(f"{corr:14s}: {copied:4d} files")

    print("Done.")


if __name__ == "__main__":
    main()
