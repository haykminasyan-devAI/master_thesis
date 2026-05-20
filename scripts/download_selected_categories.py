#!/usr/bin/env python3
"""
Download selected CO3D categories (_000 + _001 zips), extract, and optionally trim.

Example:
  python scripts/download_selected_categories.py \
    --data_root /home/asds/project_Hayk_Minasyan/data/co3d \
    --links_file /home/asds/project_Hayk_Minasyan/co3d/co3d/links.json \
    --categories "hydrant,laptop,motorcycle,parkingmeter,skateboard,toybus" \
    --keep 6
"""

import argparse
import json
import os
import shutil
import zipfile

import requests
from tqdm import tqdm


def download_file(url: str, dest_path: str) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc=os.path.basename(dest_path),
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            written = f.write(chunk)
            bar.update(written)


def extract_zip(zip_path: str, dest_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def trim_category(category_dir: str, keep: int) -> int:
    seq_with_pc = []
    seq_without_pc = []
    for entry in sorted(os.listdir(category_dir)):
        p = os.path.join(category_dir, entry)
        if not os.path.isdir(p):
            continue
        if os.path.isfile(os.path.join(p, "pointcloud.ply")):
            seq_with_pc.append(entry)
        else:
            seq_without_pc.append(entry)
    keep_n = min(keep, len(seq_with_pc))
    keep_set = set(seq_with_pc[:keep_n])
    for seq in (seq_with_pc + seq_without_pc):
        if seq not in keep_set:
            shutil.rmtree(os.path.join(category_dir, seq), ignore_errors=True)
    return keep_n


def parse_categories(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--links_file", required=True)
    ap.add_argument("--categories", required=True, help="comma-separated list")
    ap.add_argument("--keep", type=int, default=6)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    with open(args.links_file, "r") as f:
        links = json.load(f)["full"]

    categories = parse_categories(args.categories)
    os.makedirs(args.data_root, exist_ok=True)

    for cat in categories:
        if cat not in links:
            print(f"[skip] unknown category in links: {cat}")
            continue
        cat_dir = os.path.join(args.data_root, cat)
        if args.skip_existing and os.path.isdir(cat_dir) and os.path.isfile(os.path.join(cat_dir, "frame_annotations.jgz")):
            print(f"[skip] existing category: {cat}")
            continue

        urls = links[cat]
        meta_url = next((u for u in urls if u.endswith("_000.zip")), None)
        data_url = next((u for u in urls if u.endswith("_001.zip")), None)
        if not meta_url or not data_url:
            print(f"[skip] missing _000/_001 links for {cat}")
            continue

        print(f"\n=== {cat} ===")
        meta_zip = os.path.join(args.data_root, f"{cat}_000.zip")
        data_zip = os.path.join(args.data_root, f"{cat}_001.zip")

        print("download meta...")
        download_file(meta_url, meta_zip)
        print("download data...")
        download_file(data_url, data_zip)

        print("extract...")
        extract_zip(meta_zip, args.data_root)
        extract_zip(data_zip, args.data_root)

        if os.path.exists(meta_zip):
            os.remove(meta_zip)
        if os.path.exists(data_zip):
            os.remove(data_zip)

        if os.path.isdir(cat_dir):
            kept = trim_category(cat_dir, args.keep)
            print(f"kept sequences with pointcloud: {kept}")
        else:
            print(f"[warn] extracted folder missing: {cat_dir}")


if __name__ == "__main__":
    main()
