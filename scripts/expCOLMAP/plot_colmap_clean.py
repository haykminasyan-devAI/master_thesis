"""
Plot COLMAP Chamfer Distance vs Number of Frames for CLEAN (unmasked) frames,
averaged over 6 CO3D categories.

Expected directory structure:
    outputs/colmap/clean/
        teddybear_101_11758_21048/
            frames_050/metrics.txt
            frames_060/metrics.txt
            ...
        hydrant_106_12648_23157/
            ...

Usage:
    python scripts/expCOLMAP/plot_colmap_clean.py \\
        --results_root outputs/colmap/clean \\
        --output_dir   outputs/colmap/clean
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP_VALUES = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]


def read_chamfer(metrics_path: str):
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
    if not os.path.isdir(results_root):
        return [], np.array([]), np.array([]), []

    seq_dirs = [
        os.path.join(results_root, d)
        for d in sorted(os.listdir(results_root))
        if os.path.isdir(os.path.join(results_root, d))
    ]

    cd_by_n = {}
    for seq_dir in seq_dirs:
        for n in SWEEP_VALUES:
            p = os.path.join(seq_dir, f"frames_{n:03d}", "metrics.txt")
            cd = read_chamfer(p)
            if cd is not None:
                cd_by_n.setdefault(n, []).append(cd)
                print(f"  {os.path.basename(seq_dir)}  n={n:>3}  CD={cd:.5f}")

    if not cd_by_n:
        return [], np.array([]), np.array([]), []

    ns     = sorted(cd_by_n.keys())
    means  = np.array([np.mean(cd_by_n[n]) for n in ns])
    stds   = np.array([np.std(cd_by_n[n])  for n in ns])
    counts = [len(cd_by_n[n]) for n in ns]
    return ns, means, stds, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/colmap/clean")
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_root
    os.makedirs(out_dir, exist_ok=True)

    print(f"Collecting from: {args.results_root}\n")
    ns, means, stds, counts = collect(args.results_root)

    if not ns:
        print("No successful COLMAP results found yet.")
        return

    n_seqs = counts[0] if counts else 0
    print(f"\n  {'n_frames':>8}  |  {'mean CD':>9}  |  {'std CD':>9}  |  n_seqs")
    print("  " + "-" * 48)
    for n, m, s, c in zip(ns, means, stds, counts):
        print(f"  {n:>8d}  |  {m:>9.5f}  |  {s:>9.5f}  |  {c}")

    # ── plot ──────────────────────────────────────────────────────────────────
    ns_arr = np.array(ns)
    color  = "#1565C0"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    ax.plot(ns_arr, means,
            marker="o", color=color, linewidth=2.5,
            markersize=7, markerfacecolor="white", markeredgewidth=2,
            label=f"COLMAP clean  (avg over {n_seqs} sequences)",
            zorder=3)
    ax.fill_between(ns_arr, means - stds, means + stds,
                    color=color, alpha=0.15, label="±1 std across sequences",
                    zorder=2)

    # annotate last value
    ax.annotate(f"{means[-1]:.4f}",
                xy=(ns_arr[-1], means[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=color, fontweight="bold", va="center")

    ax.set_title("COLMAP (Clean Frames) — Chamfer Distance vs Number of Frames",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of input frames", fontsize=12, labelpad=8)
    ax.set_ylabel("Chamfer Distance  (↓ lower is better)", fontsize=12, labelpad=8)
    ax.set_xticks(SWEEP_VALUES)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.margins(y=0.08)

    fig.suptitle(
        "COLMAP SfM — Averaged over 6 CO3D categories\n"
        "(teddybear · hydrant · cup · bottle · toybus · toytrain)\n"
        "Clean frames (no masking)  |  Mean ± 1 std",
        fontsize=10, color="#444444", y=1.01)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "colmap_clean_chamfer_vs_nframes.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
