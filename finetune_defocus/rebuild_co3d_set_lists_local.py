#!/usr/bin/env python3
"""
Rebuild CO3D category set_lists from frames that actually exist on disk.

Official fewview_train JSONs reference sequences you may not have downloaded.
This script backs up set_lists/ and writes a single fewview_train file whose
entries are filtered from frame_annotations.j.gz to:
  - sequences listed in <category>_selected_8.json (or --seq_json), and
  - RGB + depth + mask paths that exist under --co3d_dir.

Output: <category>/set_lists/fewview_train_local_frames.json
  { "train": [[seq, frame_number, filepath], ...], "test": [...] }

Both splits list the same frames (preprocess runs train+test passes).
"""
import argparse
import gzip
import json
import os
import shutil
from pathlib import Path


def image_path_from_frame(fr: dict) -> str:
    im = fr.get("image")
    if isinstance(im, dict) and "path" in im:
        return im["path"]
    if "image_path" in fr:
        return fr["image_path"]
    if "path" in fr:
        return fr["path"]
    raise KeyError("cannot find image path in frame record")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--co3d_dir", required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument(
        "--seq_json_suffix",
        default="_selected_8.json",
        help="per-category JSON list of sequence names, e.g. apple_selected_8.json",
    )
    p.add_argument("--no_backup", action="store_true", help="do not move set_lists to set_lists.bak_*")
    args = p.parse_args()

    root = Path(args.co3d_dir)

    for cat in args.categories:
        cat_dir = root / cat
        seq_path = root / f"{cat}{args.seq_json_suffix}"
        if not seq_path.is_file():
            raise FileNotFoundError(seq_path)
        selected = set(json.loads(seq_path.read_text()))
        if len(selected) < 1:
            raise ValueError(f"{seq_path} is empty")

        frame_jgz = cat_dir / "frame_annotations.jgz"
        if not frame_jgz.is_file():
            raise FileNotFoundError(frame_jgz)

        with gzip.open(frame_jgz, "rt") as f:
            frame_data = json.load(f)

        entries = []
        for fr in frame_data:
            seq = fr["sequence_name"]
            if seq not in selected:
                continue
            fnum = fr["frame_number"]
            try:
                filepath = image_path_from_frame(fr)
            except KeyError:
                continue
            fd = fr
            mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
            depth_rel = fd["depth"]["path"]
            img_abs = root / filepath
            dep_abs = root / depth_rel
            msk_abs = root / mask_path
            if img_abs.is_file() and dep_abs.is_file() and msk_abs.is_file():
                entries.append([seq, fnum, filepath])

        if not entries:
            raise RuntimeError(
                f"{cat}: no local frame triplets for selected sequences. "
                "Download RGB/depth/mask for those sequences."
            )

        set_dir = cat_dir / "set_lists"
        if not args.no_backup and set_dir.is_dir():
            bak = cat_dir / f"set_lists.bak_{os.getpid()}"
            shutil.move(str(set_dir), str(bak))
        set_dir.mkdir(parents=True, exist_ok=True)

        payload = {"train": entries, "test": entries}
        out_file = set_dir / "fewview_train_local_frames.json"
        out_file.write_text(json.dumps(payload))
        print(f"{cat}: wrote {out_file} ({len(entries)} frame entries)")


if __name__ == "__main__":
    main()
