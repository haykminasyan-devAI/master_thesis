"""
Average COLMAP Chamfer Distance across 6 CO3D sequences and plot
frames vs Chamfer distance with ±1 std shaded band.

Expected directory structure:
    outputs/colmap/masked_25pct/
        teddybear_101_11758_21048/
            frames_02/metrics.txt
            frames_10/metrics.txt
            frames_50/metrics.txt
            ...
        hydrant_106_12648_23157/
            ...

Usage:
    python scripts/expCOLMAP/plot_colmap_results.py
    python scripts/expCOLMAP/plot_colmap_results.py \\
        --results_root outputs/colmap/masked_25pct \\
        --output_dir   outputs/colmap/masked_25pct
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP_VALUES = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]


def read_chamfer(metrics_path: str):
    """Return Chamfer distance from a metrics.txt, or None on failure/skip."""
    if not os.path.isfile(metrics_path):
        return None
    status_ok = False
    cd = None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("status:") and "ok" in line:
                status_ok = True
            if line.startswith("chamfer_distance:"):
                try:
                    cd = float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return cd if status_ok else None


def collect(results_root: str):
    """
    Scan all sequence subdirectories and collect Chamfer distances.

    Returns:
        ns     : sorted list of n_frames values that have ≥1 result
        means  : mean CD per n_frames
        stds   : std  CD per n_frames
        counts : number of sequences contributing to each point
    """
    if not os.path.isdir(results_root):
        return [], np.array([]), np.array([]), []

    seq_dirs = [
        os.path.join(results_root, d)
        for d in sorted(os.listdir(results_root))
        if os.path.isdir(os.path.join(results_root, d))
    ]

    cd_by_n: dict = {}
    for seq_dir in seq_dirs:
        for n in SWEEP_VALUES:
            p = os.path.join(seq_dir, f"frames_{n:02d}", "metrics.txt")
            cd = read_chamfer(p)
            if cd is not None:
                cd_by_n.setdefault(n, []).append(cd)

    if not cd_by_n:
        return [], np.array([]), np.array([]), []

    ns     = sorted(cd_by_n.keys())
    means  = np.array([np.mean(cd_by_n[n]) for n in ns])
    stds   = np.array([np.std(cd_by_n[n])  for n in ns])
    counts = [len(cd_by_n[n]) for n in ns]
    return ns, means, stds, counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_root", default="outputs/colmap/masked_25pct",
                        help="Root dir containing one subdir per sequence")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save the plot (default: results_root)")
    parser.add_argument("--n_min", type=int, default=50)
    parser.add_argument("--n_max", type=int, default=150)
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_root
    os.makedirs(out_dir, exist_ok=True)

    ns, means, stds, counts = collect(args.results_root)

    if not ns:
        print("No successful COLMAP results found yet — nothing to plot.")
        return

    n_seqs = counts[0] if counts else 0
    print(f"Collected {len(ns)} n_frames values  |  up to {n_seqs} sequences per point")

    # ── text summary ──────────────────────────────────────────────────────────
    print(f"\n  {'n_frames':>8}  |  {'mean CD':>9}  |  {'std CD':>9}  |  n_seqs")
    print("  " + "-" * 48)
    for n, m, s, c in zip(ns, means, stds, counts):
        print(f"  {n:>8d}  |  {m:>9.5f}  |  {s:>9.5f}  |  {c}")

    # ── plot ──────────────────────────────────────────────────────────────────
    ns_arr = np.array(ns)
    color  = "#1565C0"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ns_arr, means,
            marker="o", color=color, linewidth=2,
            markersize=5, markerfacecolor="white", markeredgewidth=1.5,
            label=f"COLMAP  (mask 25%, 3 patches,  n_seq={n_seqs})")
    ax.fill_between(ns_arr, means - stds, means + stds,
                    color=color, alpha=0.15, label="±1 std across sequences")

    ax.set_title("COLMAP — Chamfer Distance vs Number of Frames",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of input frames", fontsize=11)
    ax.set_ylabel("Chamfer Distance  (↓ better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "COLMAP SfM — Averaged over 6 CO3D categories\n"
        "(teddybear · hydrant · cup · bottle · toybus · toytrain)\n"
        "Masked frames: ratio=0.25, 3 patches, seed=42  |  Mean ± 1 std",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, "colmap_chamfer_vs_nframes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
