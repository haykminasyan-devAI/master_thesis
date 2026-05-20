#!/usr/bin/env python3
"""
Pick one CO3D sequence per category for the 10-category fine-tuning setup.
Writes sequences_10cat.json: { "apple": "seq_id", ... }

Scans raw CO3D layout: <co3d_dir>/<category>/<sequence_name>/images/frame*.jpg
Chooses the first sequence (lexicographic) that has at least one frame.
Override with --manifest if you want to pin IDs yourself.
"""

import argparse
import glob
import json
import os
from typing import Optional

CATEGORIES_10 = [
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
]


def discover_one_sequence(co3d_dir: str, category: str) -> Optional[str]:
    cat_dir = os.path.join(co3d_dir, category)
    if not os.path.isdir(cat_dir):
        return None
    candidates = []
    for name in sorted(os.listdir(cat_dir)):
        if name in ("set_lists",):
            continue
        seq_dir = os.path.join(cat_dir, name)
        if not os.path.isdir(seq_dir):
            continue
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            continue
        jpgs = glob.glob(os.path.join(img_dir, "frame*.jpg"))
        if jpgs:
            candidates.append(name)
    return candidates[0] if candidates else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--co3d_dir", required=True, help="Raw CO3D root (contains apple/, banana/, ...)")
    p.add_argument("--out", default="finetune_blur/sequences_10cat.json")
    p.add_argument("--categories", nargs="*", default=CATEGORIES_10)
    args = p.parse_args()

    mapping = {}
    missing = []
    for cat in args.categories:
        sid = discover_one_sequence(args.co3d_dir, cat)
        if sid is None:
            missing.append(cat)
            print(f"ERROR: no sequence with images under {args.co3d_dir}/{cat}/")
        else:
            mapping[cat] = sid
            print(f"  {cat}: {sid}")

    if missing:
        print(f"\nFAIL: missing data for {len(missing)} categories: {', '.join(missing)}")
        print("Download CO3D raw data for these categories into CO3D_ROOT first.")
        print("Example (from repo co3d/co3d, comma-separated):")
        print(
            "  python download_dataset.py --download_folder <CO3D_ROOT> \\\n"
            "    --download_categories "
            "apple,banana,baseballbat,baseballglove,bicycle,bowl,broccoli,cake,car,carrot"
        )
        return 1

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"Written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
