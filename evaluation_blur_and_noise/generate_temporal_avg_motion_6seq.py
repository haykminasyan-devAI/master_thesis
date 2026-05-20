#!/usr/bin/env python3
"""
GoPro-style motion blur: average of W consecutive frames (temporal box filter).
For each time t, blur[t] = mean( images[t-k] ... images[t+k] ) with k=(W-1)//2,
index clamped to sequence edges. Neighbor frames are resized to the central frame
size (handles mixed resolutions in CO3D preprocessed data).

Layout:
  co3d_root / <category> / <seq_id> / images / frame*.jpg
  -> motion_root / <category> / <seq_id> / <motion_tag> / frame*.jpg

Default category/seq pairs match sequences_6eval.inc.sh (keep in sync).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Must match evaluation_blur_and_noise/sequences_6eval.inc.sh
DEFAULT_SEQUENCES: dict[str, str] = {
    "bottle": "34_1397_4376",
    "cup": "12_100_593",
    "donut": "110_13050_22740",
    "teddybear": "101_11758_21048",
    "couch": "105_12576_23188",
    "toytrain": "104_12352_22039",
}


def list_frames(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_p = [p for p in images_dir.iterdir() if p.suffix in exts]
    all_p.sort(
        key=lambda p: (
            int(m.group(1)) if (m := re.search(r"(\d+)", p.stem)) else p.stem
        )
    )
    if not all_p:
        raise FileNotFoundError(f"No images in {images_dir}")
    return all_p


def process_sequence(
    co3d_root: Path,
    motion_root: Path,
    category: str,
    seq_id: str,
    motion_tag: str,
    window: int,
    dry_run: bool,
    skip_if_complete: bool = True,
) -> int:
    images_dir = co3d_root / category / seq_id / "images"
    if not images_dir.is_dir():
        print(f"SKIP (no images): {images_dir}", file=sys.stderr)
        return 0
    out_dir = motion_root / category / seq_id / motion_tag
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = list_frames(images_dir)
    n = len(paths)
    if skip_if_complete and not dry_run:
        existing = sum(1 for p in paths if (out_dir / p.name).exists())
        if existing == n:
            print(f"{category}/{seq_id}: already complete ({n} frames) -> {out_dir}")
            return 0
        if existing:
            print(f"{category}/{seq_id}: resuming, {existing}/{n} already exist -> {out_dir}")
    half = (window - 1) // 2
    print(f"{category}/{seq_id}: {n} frames, W={window} -> {out_dir}")

    written = 0
    for t in range(n):
        out_name = paths[t].name
        out_path = out_dir / out_name
        if skip_if_complete and (not dry_run) and out_path.exists():
            continue
        # central size for this output frame
        ref = np.array(Image.open(paths[t]).convert("RGB"))
        h, w = ref.shape[:2]
        acc = np.zeros((h, w, 3), dtype=np.float32)
        count = 0
        for k in range(-half, half + 1):
            j = min(max(t + k, 0), n - 1)
            im = np.array(Image.open(paths[j]).convert("RGB"), dtype=np.float32)
            if im.shape[0] != h or im.shape[1] != w:
                pil = Image.fromarray(im.clip(0, 255).astype(np.uint8))
                pil = pil.resize((w, h), Image.BILINEAR)
                im = np.array(pil, dtype=np.float32)
            acc += im
            count += 1
        acc /= max(1, count)
        out = np.clip(acc, 0, 255).astype(np.uint8)
        if not dry_run:
            Image.fromarray(out).save(out_path, quality=95)
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--co3d_root", required=True, type=Path, help="Preprocessed CO3D root (with images/ per sequence)")
    ap.add_argument("--motion_root", required=True, type=Path, help="Output root for motion-blurred trees")
    ap.add_argument(
        "--motion_tag",
        default="temporal_avg_w11_gopro_like",
        help="Subfolder name under each sequence (match finetune Motion tag)",
    )
    ap.add_argument("--window", type=int, default=11, help="Temporal window (odd, e.g. 11)")
    ap.add_argument("--dry_run", action="store_true", help="Only print; do not write")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of categories (default: all 6 in DEFAULT_SEQUENCES)",
    )
    ap.add_argument(
        "--seq_json",
        nargs="*",
        default=None,
        help=(
            "One or more selected_seqs_<name>.json files (CO3D processed format: "
            "{category: {seq_id: [...]}}). When set, all (cat, seq_id) pairs from "
            "these files override DEFAULT_SEQUENCES."
        ),
    )
    ap.add_argument(
        "--no_skip_complete",
        action="store_true",
        help="Re-render even if every output file already exists (default: skip).",
    )
    args = ap.parse_args()
    w = int(args.window)
    if w < 1 or w % 2 == 0:
        ap.error("--window must be a positive odd integer")

    if args.seq_json:
        pairs: list[tuple[str, str]] = []
        seen = set()
        for jp in args.seq_json:
            data = json.loads(Path(jp).read_text())
            for cat, scenes in data.items():
                if not isinstance(scenes, dict):
                    continue
                for sid in scenes.keys():
                    key = (cat, sid)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append(key)
        if args.only:
            pairs = [(c, s) for (c, s) in pairs if c in set(args.only)]
        pairs.sort()
        if not pairs:
            ap.error("No (category, seq_id) pairs found from --seq_json (after --only filter)")
        print(f"Loaded {len(pairs)} sequences from {len(args.seq_json)} JSON file(s).")
    else:
        seqs = {k: v for k, v in DEFAULT_SEQUENCES.items()}
        if args.only:
            seqs = {k: seqs[k] for k in args.only if k in seqs}
            missing = set(args.only) - set(seqs.keys())
            if missing:
                ap.error(f"Unknown categories: {missing}")
        pairs = sorted(seqs.items())

    total = 0
    for cat, sid in pairs:
        n = process_sequence(
            args.co3d_root.resolve(),
            args.motion_root.resolve(),
            cat,
            sid,
            args.motion_tag,
            w,
            args.dry_run,
            skip_if_complete=not args.no_skip_complete,
        )
        total += n
    print(f"Done. Wrote {total} frames total (dry_run={args.dry_run})")


if __name__ == "__main__":
    main()
