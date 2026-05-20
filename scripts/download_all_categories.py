"""
Download all 50 CO3D categories one at a time, keeping only N sequences
with GT point clouds per category to stay within disk space limits.

For each category:
  1. Download _000.zip (metadata, ~30-50 MB)
  2. Download _001.zip (~20 GB image sequences)
  3. Extract both zips
  4. Keep only `keep_per_category` sequences that have pointcloud.ply
  5. Delete all other sequences + zip files
  6. Move to next category

Usage:
    python download_all_categories.py \
        --data_root /home/asds/project_Hayk_Minasyan/data/co3d \
        --links_file /home/asds/project_Hayk_Minasyan/co3d/co3d/links.json \
        --keep 6
"""

import os
import json
import shutil
import zipfile
import argparse
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm


# Categories already processed — will be skipped for download but still trimmed
ALREADY_DOWNLOADED = {"apple", "teddybear", "tv", "microwave", "parkingmeter", "baseballglove", "baseballbat"}


def download_file(url: str, dest_path: str) -> None:
    print(f"  Downloading {os.path.basename(dest_path)} ...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="iB", unit_scale=True, unit_divisor=1024,
        desc=os.path.basename(dest_path)
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            size = f.write(chunk)
            bar.update(size)


def extract_zip(zip_path: str, dest_dir: str) -> None:
    print(f"  Extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  Extracted.")


def trim_category(category_dir: str, keep: int) -> int:
    sequences_with_pc = []
    sequences_without_pc = []

    for entry in sorted(os.listdir(category_dir)):
        entry_path = os.path.join(category_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if os.path.isfile(os.path.join(entry_path, "pointcloud.ply")):
            sequences_with_pc.append(entry)
        else:
            sequences_without_pc.append(entry)

    actual_keep = min(keep, len(sequences_with_pc))
    to_keep = set(sequences_with_pc[:actual_keep])
    to_delete = [s for s in (sequences_with_pc + sequences_without_pc) if s not in to_keep]

    for seq in to_delete:
        shutil.rmtree(os.path.join(category_dir, seq))

    print(f"  Kept {actual_keep} sequences with pointcloud.ply, deleted {len(to_delete)}.")
    return actual_keep


def process_category(category: str, urls: list, data_root: str, keep: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Category: {category}")
    print(f"{'='*60}")

    category_dir = os.path.join(data_root, category)

    if category in ALREADY_DOWNLOADED:
        print(f"  Already downloaded. Checking if trimming is needed ...")
        if os.path.isdir(category_dir):
            trim_category(category_dir, keep)
        return

    # Find metadata zip (_000) and first data zip (_001)
    meta_url = next((u for u in urls if u.endswith("_000.zip")), None)
    data_url = next((u for u in urls if u.endswith("_001.zip")), None)

    if not meta_url or not data_url:
        print(f"  WARNING: Could not find _000.zip or _001.zip for {category}, skipping.")
        return

    meta_zip = os.path.join(data_root, f"{category}_000.zip")
    data_zip = os.path.join(data_root, f"{category}_001.zip")

    # Download
    download_file(meta_url, meta_zip)
    download_file(data_url, data_zip)

    # Extract
    extract_zip(meta_zip, data_root)
    extract_zip(data_zip, data_root)

    # Delete zips immediately to free space
    os.remove(meta_zip)
    os.remove(data_zip)
    print(f"  Deleted zip files.")

    # Trim to keep only N sequences with pointcloud.ply
    if os.path.isdir(category_dir):
        trim_category(category_dir, keep)
    else:
        print(f"  WARNING: Category folder {category_dir} not found after extraction.")

    # Log progress
    log_path = os.path.join(data_root, "download_progress.log")
    with open(log_path, "a") as log:
        log.write(f"{category}: done\n")

    print(f"  Category {category} complete.")


def main():
    parser = argparse.ArgumentParser(description="Download all CO3D categories with disk-space-aware trimming.")
    parser.add_argument("--data_root", required=True, help="Path to CO3D data root directory")
    parser.add_argument("--links_file", required=True, help="Path to CO3D links.json file")
    parser.add_argument("--keep", type=int, default=6, help="Number of sequences to keep per category (default: 6)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from this category name")
    args = parser.parse_args()

    with open(args.links_file, "r") as f:
        links = json.load(f)["full"]

    categories = [c for c in links.keys() if c != "METADATA"]
    print(f"Total categories to process: {len(categories)}")

    resume = args.resume_from is None
    completed = 0
    total_to_process = sum(1 for c in categories if c not in ALREADY_DOWNLOADED)
    if args.resume_from:
        idx = categories.index(args.resume_from) if args.resume_from in categories else 0
        total_to_process = len(categories) - idx
    overall_start = time.time()

    for i, category in enumerate(categories):
        if not resume:
            if category == args.resume_from:
                resume = True
            else:
                print(f"  Skipping {category} (before resume point)")
                continue

        cat_start = time.time()
        process_category(category, links[category], args.data_root, args.keep)
        cat_elapsed = time.time() - cat_start

        if category not in ALREADY_DOWNLOADED:
            completed += 1
            remaining = total_to_process - completed
            avg_time = (time.time() - overall_start) / completed
            eta = timedelta(seconds=int(avg_time * remaining))
            print(f"\n  Progress: {completed}/{total_to_process} categories done")
            print(f"  Time for this category: {timedelta(seconds=int(cat_elapsed))}")
            print(f"  Estimated time remaining: {eta}")
            print(f"  Estimated finish: {datetime.now() + eta:%Y-%m-%d %H:%M}")

    print("\n\nAll categories processed!")


if __name__ == "__main__":
    main()
