#!/usr/bin/env python3
"""
Create selected_seqs_train.json and selected_seqs_test.json for the 10-category
setup, using sequences from sequences_10cat.json (from discover_sequences_10cat.py).

Run AFTER preprocessing into co3d_processed (see preprocess_for_training_10cat.sh).
"""

import argparse
import glob
import json
import os

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


def get_frame_indices(root, category, seq_id):
    img_dir = os.path.join(root, category, seq_id, "images")
    jpgs = sorted(glob.glob(os.path.join(img_dir, "frame*.jpg")))
    return [int(os.path.basename(j)[5:-4]) for j in jpgs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--co3d_processed", required=True)
    p.add_argument(
        "--sequences_json",
        required=True,
        help="Path to sequences_10cat.json from discover_sequences_10cat.py",
    )
    args = p.parse_args()

    with open(args.sequences_json) as f:
        sequences = json.load(f)

    all_seqs = {}
    for cat in CATEGORIES_10:
        if cat not in sequences:
            print(f"WARNING: category {cat} missing from {args.sequences_json}")
            continue
        seq_id = sequences[cat]
        frames = get_frame_indices(args.co3d_processed, cat, seq_id)
        if not frames:
            print(f"WARNING: no preprocessed frames for {cat}/{seq_id}")
        else:
            print(f"  {cat}/{seq_id}: {len(frames)} frames")
            all_seqs.setdefault(cat, {})[seq_id] = frames

    if not all_seqs:
        print("ERROR: no sequences found.")
        return 1

    for split in ("train", "test"):
        out = os.path.join(args.co3d_processed, f"selected_seqs_{split}.json")
        with open(out, "w") as f:
            json.dump(all_seqs, f, indent=2)
        print(f"Written: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
