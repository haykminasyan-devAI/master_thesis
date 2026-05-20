#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


DEFAULT_BASE10 = [
    "apple",
    "banana",
    "baseballbat",
    "baseballglove",
    "bicycle",
    "broccoli",
    "bowl",
    "cake",
    "car",
    "carrot",
]

DEFAULT_TEST6 = ["cup", "couch", "bottle", "teddybear", "donut", "toytrain"]
DEFAULT_VAL3 = ["laptop", "tv", "handbag"]
PREFERRED_POOL = [
    "laptop",
    "tv",
    "handbag",
    "frisbee",
    "hairdryer",
    "hotdog",
    "hydrant",
    "kite",
    "microwave",
    "motorcycle",
    "parkingmeter",
    "pizza",
    "sandwich",
    "skateboard",
    "stopsign",
    "toaster",
    "toybus",
    "toyplane",
    "wineglass",
]


def has_sequence_payload(seq_dir: Path) -> bool:
    return (seq_dir / "images").is_dir()


def list_sequences(cat_dir: Path):
    if not cat_dir.is_dir():
        return []
    seqs = [d.name for d in sorted(cat_dir.iterdir()) if d.is_dir() and has_sequence_payload(d)]
    return seqs


def copy_seq(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def main():
    ap = argparse.ArgumentParser("Build extra-10-category x 8-sequence processed split")
    ap.add_argument("--source_root", required=True, help="Processed CO3D root with many categories")
    ap.add_argument("--target_root", required=True, help="Output processed root to write extra categories")
    ap.add_argument("--n_categories", type=int, default=10)
    ap.add_argument("--n_sequences", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--manifest_out", default=None, help="Where to write selected manifest json")
    args = ap.parse_args()

    src_root = Path(args.source_root)
    dst_root = Path(args.target_root)
    if not src_root.is_dir():
        raise FileNotFoundError(f"source_root not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    excluded = set(DEFAULT_BASE10 + DEFAULT_TEST6)
    mandatory = list(DEFAULT_VAL3)
    picked = []

    def qualifies(cat: str):
        if cat in excluded:
            return False
        seqs = list_sequences(src_root / cat)
        return len(seqs) >= args.n_sequences

    # 1) Pick mandatory val categories first if available.
    for cat in mandatory:
        if cat in picked:
            continue
        if qualifies(cat):
            picked.append(cat)

    # 2) Then preferred pool.
    for cat in PREFERRED_POOL:
        if len(picked) >= args.n_categories:
            break
        if cat in picked:
            continue
        if qualifies(cat):
            picked.append(cat)

    # 3) Fallback: any qualifying category in source root.
    if len(picked) < args.n_categories:
        for cat_dir in sorted(src_root.iterdir()):
            if len(picked) >= args.n_categories:
                break
            if not cat_dir.is_dir():
                continue
            cat = cat_dir.name
            if cat in picked:
                continue
            if qualifies(cat):
                picked.append(cat)

    if len(picked) < args.n_categories:
        src_cats = sorted([d.name for d in src_root.iterdir() if d.is_dir()])
        hint = ""
        if set(src_cats).issubset(set(DEFAULT_BASE10)):
            hint = (
                " Hint: source_root looks like base 10cat only. "
                "Use a richer processed source (with extra categories), "
                "and keep target_root as co3d_processed_10cat8seq_fixed."
            )
        raise RuntimeError(
            f"Could not find {args.n_categories} categories with >= {args.n_sequences} sequences. "
            f"Found only {len(picked)}: {picked}.{hint}"
        )

    manifest = {}
    for cat in picked:
        seqs = list_sequences(src_root / cat)[: args.n_sequences]
        manifest[cat] = seqs
        for sid in seqs:
            src_seq = src_root / cat / sid
            dst_seq = dst_root / cat / sid
            if args.dry_run:
                print(f"[DRY] copy {src_seq} -> {dst_seq}")
            else:
                copy_seq(src_seq, dst_seq)

    out = args.manifest_out or str(dst_root / "selected_seqs_extra10_8.json")
    if not args.dry_run:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print("Selected categories:", picked)
    print("Manifest:", out)


if __name__ == "__main__":
    main()
