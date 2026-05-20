#!/usr/bin/env python3
"""Create sequence-level selected_seqs_<split>.json files for 10cat x 8seq setup.

Inputs:
  - processed_root/selected_seqs_train.json and selected_seqs_test.json: per-sequence
    frame-id mappings {category: {seq_id: frame_ids}}. By default both are merged so
    sequences that only appear in CO3D's official test split are still available for
    the 8-seq experiment (counts must sum to 8, e.g. 6/1/1 or 6/2/0 — random shuffle of the 8 ids).
  - raw_root/<category>_selected_8.json: exactly 8 chosen sequence ids per category

Outputs (under processed_root):
  - selected_seqs_<train_name>.json
  - selected_seqs_<val_name>.json
  - selected_seqs_<test_name>.json
"""

import argparse
import json
import os
import random


def parse_args():
    p = argparse.ArgumentParser("Prepare sequence-level 10cat x 8seq split files")
    p.add_argument("--processed_root", required=True)
    p.add_argument("--raw_root", required=True)
    p.add_argument(
        "--categories",
        nargs="+",
        default=[
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
        ],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train", type=int, default=6)
    p.add_argument("--n_val", type=int, default=1)
    p.add_argument("--n_test", type=int, default=1)
    p.set_defaults(merge_train_test_source=True)
    p.add_argument(
        "--no_merge_train_test_source",
        action="store_false",
        dest="merge_train_test_source",
        help="Use only --source_split JSON (default: merge train+test processed maps)",
    )
    p.add_argument(
        "--source_split",
        default="train",
        help="When merge is off: read selected_seqs_<source_split>.json only",
    )
    p.add_argument("--train_name", default="train_10cat8")
    p.add_argument("--val_name", default="val_10cat8")
    p.add_argument("--test_name", default="test_10cat8")
    return p.parse_args()


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _merge_processed_sources(processed_root, categories):
    train_path = os.path.join(processed_root, "selected_seqs_train.json")
    test_path = os.path.join(processed_root, "selected_seqs_test.json")
    mt = _load_json(train_path)
    ms = _load_json(test_path)
    merged = {}
    for cat in categories:
        a = mt.get(cat) if isinstance(mt.get(cat), dict) else {}
        b = ms.get(cat) if isinstance(ms.get(cat), dict) else {}
        if a is None:
            a = {}
        if b is None:
            b = {}
        overlap = set(a) & set(b)
        if overlap:
            for sid in overlap:
                if a[sid] != b[sid]:
                    raise ValueError(
                        f"{cat}/{sid}: train and test processed frame lists differ; "
                        "cannot merge safely."
                    )
        merged[cat] = {**a, **b}
    return merged


def main():
    args = parse_args()
    if args.n_train + args.n_val + args.n_test != 8:
        raise ValueError("Expected n_train + n_val + n_test == 8 for selected_8 setup")

    if args.merge_train_test_source:
        source = _merge_processed_sources(args.processed_root, args.categories)
    else:
        source_path = os.path.join(args.processed_root, f"selected_seqs_{args.source_split}.json")
        source = _load_json(source_path)

    out_train, out_val, out_test = {}, {}, {}
    for cat_idx, cat in enumerate(args.categories):
        sel_path = os.path.join(args.raw_root, f"{cat}_selected_8.json")
        selected = _load_json(sel_path)
        if not isinstance(selected, list) or len(selected) != 8:
            raise ValueError(f"{sel_path} must contain exactly 8 sequence ids")
        if cat not in source:
            raise KeyError(f"{cat} missing in {source_path}")

        missing = [sid for sid in selected if sid not in source[cat]]
        if missing:
            raise KeyError(f"{cat}: selected sequence(s) missing in processed source mapping: {missing}")

        seqs = list(selected)
        rng = random.Random(args.seed + cat_idx)
        rng.shuffle(seqs)

        train_ids = seqs[: args.n_train]
        val_ids = seqs[args.n_train : args.n_train + args.n_val]
        test_ids = seqs[args.n_train + args.n_val :]

        out_train[cat] = {sid: source[cat][sid] for sid in train_ids}
        out_val[cat] = {sid: source[cat][sid] for sid in val_ids}
        out_test[cat] = {sid: source[cat][sid] for sid in test_ids}

    train_path = os.path.join(args.processed_root, f"selected_seqs_{args.train_name}.json")
    val_path = os.path.join(args.processed_root, f"selected_seqs_{args.val_name}.json")
    test_path = os.path.join(args.processed_root, f"selected_seqs_{args.test_name}.json")
    _dump_json(train_path, out_train)
    _dump_json(val_path, out_val)
    _dump_json(test_path, out_test)

    print("Wrote:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print("Per-category sequence counts:")
    for cat in args.categories:
        print(
            f"  {cat:15s} train={len(out_train[cat])} "
            f"val={len(out_val[cat])} test={len(out_test[cat])}"
        )


if __name__ == "__main__":
    main()

