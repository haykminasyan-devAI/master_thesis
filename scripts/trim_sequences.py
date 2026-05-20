"""
Trim a CO3D category directory to keep only N sequences that have a pointcloud.ply.
Usage:
    python trim_sequences.py --data_root /path/to/co3d --categories apple teddybear --keep 6
"""

import os
import shutil
import argparse


def trim_category(category_dir: str, keep: int) -> None:
    category = os.path.basename(category_dir)

    # Collect all sequence folders that contain a pointcloud.ply
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

    print(f"\n[{category}]")
    print(f"  Total sequences:          {len(sequences_with_pc) + len(sequences_without_pc)}")
    print(f"  With pointcloud.ply:      {len(sequences_with_pc)}")
    print(f"  Without pointcloud.ply:   {len(sequences_without_pc)}")

    if len(sequences_with_pc) < keep:
        print(f"  WARNING: Only {len(sequences_with_pc)} sequences with pointcloud, "
              f"cannot keep {keep}. Keeping all {len(sequences_with_pc)}.")
        keep = len(sequences_with_pc)

    to_keep = set(sequences_with_pc[:keep])
    to_delete = [s for s in (sequences_with_pc + sequences_without_pc) if s not in to_keep]

    print(f"  Keeping:  {sorted(to_keep)}")
    print(f"  Deleting: {len(to_delete)} sequences ...")

    for seq in to_delete:
        seq_path = os.path.join(category_dir, seq)
        shutil.rmtree(seq_path)

    print(f"  Done. {keep} sequences remaining.")


def main():
    parser = argparse.ArgumentParser(description="Trim CO3D category to N sequences with GT point clouds.")
    parser.add_argument("--data_root", required=True, help="Path to CO3D data root (contains apple/, teddybear/, etc.)")
    parser.add_argument("--categories", nargs="+", required=True, help="Category names to trim")
    parser.add_argument("--keep", type=int, default=6, help="Number of sequences to keep per category (default: 6)")
    args = parser.parse_args()

    for category in args.categories:
        category_dir = os.path.join(args.data_root, category)
        if not os.path.isdir(category_dir):
            print(f"WARNING: {category_dir} does not exist, skipping.")
            continue
        trim_category(category_dir, args.keep)

    print("\nAll done.")


if __name__ == "__main__":
    main()
