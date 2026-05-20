#!/usr/bin/env python3
"""
Build train/val sequence splits with ~7:1 ratio (train holds ~7/8 of sequences per category).

Reads selected_seqs_<src_split>.json (category -> {seq_id: frame_indices}),
writes selected_seqs_<out_train>.json and selected_seqs_<out_val>.json next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d_root", type=Path, required=True)
    ap.add_argument("--src_split", type=str, default="train_10cat8", help="reads selected_seqs_{src_split}.json")
    ap.add_argument("--out_train_suffix", type=str, default="train_10cat8_7v1")
    ap.add_argument("--out_val_suffix", type=str, default="val_10cat8_7v1")
    args = ap.parse_args()

    src_path = args.co3d_root / f"selected_seqs_{args.src_split}.json"
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    data = json.loads(src_path.read_text())
    train_out: dict = {}
    val_out: dict = {}

    for cat, seqs in data.items():
        if not isinstance(seqs, dict) or not seqs:
            continue
        keys = sorted(seqs.keys())
        n = len(keys)
        # ~1/8 validation sequences (7:1 train:val count), at least 1 val when possible
        if n <= 1:
            train_out[cat] = dict(seqs)
            continue
        n_val = max(1, n // 8)
        if n_val >= n:
            n_val = n - 1
        val_keys = keys[-n_val:]
        train_keys = keys[:-n_val]
        train_out[cat] = {k: seqs[k] for k in train_keys}
        val_out[cat] = {k: seqs[k] for k in val_keys}

    train_path = args.co3d_root / f"selected_seqs_{args.out_train_suffix}.json"
    val_path = args.co3d_root / f"selected_seqs_{args.out_val_suffix}.json"
    train_path.write_text(json.dumps(train_out, indent=2))
    val_path.write_text(json.dumps(val_out, indent=2))
    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")
    for cat in sorted(train_out.keys()):
        nt = len(train_out[cat])
        nv = len(val_out.get(cat, {}))
        print(f"  {cat}: train_seq={nt} val_seq={nv}")


if __name__ == "__main__":
    main()
