#!/usr/bin/env python3
"""After merge_co3d_root_selected_seqs + implicit train+test merge, check all 8 ids exist per category."""
import argparse
import json
import os

DEFAULT_10 = [
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_root", required=True)
    p.add_argument("--raw_root", required=True)
    p.add_argument("--categories", nargs="+", default=DEFAULT_10)
    args = p.parse_args()
    tr = json.load(open(os.path.join(args.processed_root, "selected_seqs_train.json")))
    te = json.load(open(os.path.join(args.processed_root, "selected_seqs_test.json")))
    ok = True
    for cat in args.categories:
        sel = json.load(open(os.path.join(args.raw_root, f"{cat}_selected_8.json")))
        if len(sel) != 8:
            print(f"BAD {cat}: expected 8 ids in *_selected_8.json, got {len(sel)}")
            ok = False
            continue
        merged = {**(tr.get(cat) or {}), **(te.get(cat) or {})}
        missing = [s for s in sel if s not in merged]
        if missing:
            print(f"BAD {cat}: missing {len(missing)} sequences after train+test merge: {missing}")
            ok = False
        else:
            print(f"OK  {cat}: all 8 selected sequences present in processed data ({len(merged)} seq keys total)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
