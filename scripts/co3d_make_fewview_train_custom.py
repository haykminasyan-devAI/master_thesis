#!/usr/bin/env python3
"""
Create CO3D set_lists/fewview_train_custom.json from local frame_annotations.jgz
when official set_lists/ is missing (partial CO3D download).

Used by dust3r/datasets_preprocess/preprocess_co3d.py (matches *fewview_train*).
"""
import argparse
import gzip
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d_dir", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--num_sequences", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--require_files",
        action="store_true",
        help="Only include frames whose image (and depth path from annotation) exist on disk",
    )
    args = ap.parse_args()

    cat_dir = os.path.join(args.co3d_dir, args.category)
    frame_file = os.path.join(cat_dir, "frame_annotations.jgz")
    if not os.path.isfile(frame_file):
        raise FileNotFoundError(frame_file)

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())

    by_seq = {}
    for f in frame_data:
        seq = f.get("sequence_name")
        fn = f.get("frame_number")
        ip = (f.get("image") or {}).get("path")
        if seq is None or fn is None or not ip:
            continue
        dp = (f.get("depth") or {}).get("path")
        image_path = os.path.join(args.co3d_dir, ip)
        if args.require_files:
            if not os.path.isfile(image_path):
                continue
            if dp:
                depth_path = os.path.join(args.co3d_dir, dp)
                if not os.path.isfile(depth_path):
                    continue
            mp = ip.replace("images", "masks").replace(".jpg", ".png")
            mask_path = os.path.join(args.co3d_dir, mp)
            if not os.path.isfile(mask_path):
                continue
        by_seq.setdefault(seq, []).append((fn, ip))

    seqs = sorted(by_seq.keys())
    if not seqs:
        raise RuntimeError(f"{args.category}: no usable frames in annotations")

    rng = random.Random(args.seed)
    if len(seqs) > args.num_sequences:
        rng.shuffle(seqs)
        seqs = sorted(seqs[: args.num_sequences])

    keep = set(seqs)
    entries = []
    for f in frame_data:
        seq_name = f.get("sequence_name")
        if seq_name not in keep:
            continue
        frame_number = f.get("frame_number")
        image_path = (f.get("image") or {}).get("path")
        if image_path is None:
            continue
        entries.append([seq_name, frame_number, image_path])

    entries.sort(key=lambda x: (x[0], x[1]))
    out_dir = os.path.join(cat_dir, "set_lists")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fewview_train_custom.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"train": entries, "test": entries}, f)

    print(
        f"[{args.category}] wrote {out_path} | sequences={len(keep)} | frames={len(entries)}"
    )


if __name__ == "__main__":
    main()
