#!/usr/bin/env python3
"""Build root selected_seqs_train.json and selected_seqs_test.json from per-category files."""
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
    p.add_argument("--categories", nargs="+", default=DEFAULT_10)
    args = p.parse_args()
    root = args.processed_root
    for split in ("train", "test"):
        merged = {}
        for c in args.categories:
            path = os.path.join(root, c, f"selected_seqs_{split}.json")
            with open(path) as f:
                merged[c] = json.load(f)
        out = os.path.join(root, f"selected_seqs_{split}.json")
        with open(out, "w") as f:
            json.dump(merged, f, indent=2)
        print("Wrote", out, "categories", list(merged.keys()))


if __name__ == "__main__":
    main()
