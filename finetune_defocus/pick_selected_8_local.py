#!/usr/bin/env python3
"""Write <category>_selected_8.json using only sequences that have on-disk RGB+depth+mask.

Use when your current *_selected_8.json lists scenes that are not fully downloaded:
preprocess will never emit those sequence keys, and prepare_seqlevel_split will fail.

Picks the N sequences with the largest number of valid frame triplets (tie-break by name).
"""
import argparse
import gzip
import json
import os
import shutil
from collections import defaultdict


def image_path_from_frame(fr: dict) -> str:
    im = fr.get("image")
    if isinstance(im, dict) and "path" in im:
        return im["path"]
    if "image_path" in fr:
        return fr["image_path"]
    if "path" in fr:
        return fr["path"]
    raise KeyError("cannot find image path in frame record")


def count_valid_frames(co3d_dir: str, cat: str):
    root = co3d_dir
    cat_dir = os.path.join(root, cat)
    frame_jgz = os.path.join(cat_dir, "frame_annotations.jgz")
    with gzip.open(frame_jgz, "rt") as f:
        frame_data = json.load(f)
    per_seq = defaultdict(int)
    for fr in frame_data:
        seq = fr["sequence_name"]
        try:
            filepath = image_path_from_frame(fr)
        except KeyError:
            continue
        fd = fr
        mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
        depth_rel = fd["depth"]["path"]
        img_abs = os.path.join(root, filepath)
        dep_abs = os.path.join(root, depth_rel)
        msk_abs = os.path.join(root, mask_path)
        if os.path.isfile(img_abs) and os.path.isfile(dep_abs) and os.path.isfile(msk_abs):
            per_seq[seq] += 1
    return per_seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--co3d_dir", required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--n_seq", type=int, default=8)
    p.add_argument("--min_frames", type=int, default=1)
    p.add_argument(
        "--exclude",
        nargs="*",
        default=(),
        help="sequence names to skip (e.g. scenes that pass file checks but fail preprocess)",
    )
    p.add_argument("--backup", action="store_true", help="save existing *_selected_8.json as .bak")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    exclude = set(args.exclude)

    for cat in args.categories:
        per_seq = count_valid_frames(args.co3d_dir, cat)
        ok = [
            (s, c)
            for s, c in per_seq.items()
            if c >= args.min_frames and s not in exclude
        ]
        ok.sort(key=lambda x: (-x[1], x[0]))
        if len(ok) < args.n_seq:
            raise RuntimeError(
                f"{cat}: only {len(ok)} sequences have >= {args.min_frames} valid RGB+depth+mask frames "
                f"(need {args.n_seq}). Download more CO3D data or lower --min_frames."
            )
        chosen = [s for s, _ in ok[: args.n_seq]]
        out_path = os.path.join(args.co3d_dir, f"{cat}_selected_8.json")
        if args.dry_run:
            print(f"{cat} DRY-RUN would write {out_path}: {chosen}")
            for s in chosen:
                print(f"  {s}: {per_seq[s]} frames")
            continue
        if args.backup and os.path.isfile(out_path):
            bak = out_path + ".bak"
            shutil.copy2(out_path, bak)
            print(f"{cat}: backed up to {bak}")
        with open(out_path, "w") as f:
            json.dump(chosen, f, indent=2)
        print(f"{cat}: wrote {out_path}")
        for s in chosen:
            print(f"  {s}: {per_seq[s]} frames")


if __name__ == "__main__":
    main()
