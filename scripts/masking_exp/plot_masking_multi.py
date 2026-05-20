"""
Average Chamfer Distance across 6 CO3D sequences and compare mask_ratio = 25% vs 50%.

Directory structure expected:
    outputs/dust3r/mask_multi/
        mask_25pct/
            teddybear_101_11758_21048/frames_02/metrics.txt
            ...
            hydrant_106_12648_23157/frames_02/metrics.txt
            ...
        mask_50pct/
            ...

For each mask tag and each n_frames:
  - collects Chamfer distance from every sequence that has results
  - computes mean ± std across sequences
  - plots mean curve with shaded ±1 std band

Produces one figure:
  masking_compare.png — mask_ratio = 25%, 50%

Usage:
    python scripts/masking_exp/plot_masking_multi.py \\
        --results_root outputs/dust3r/mask_multi \\
        --n_min 2 --n_max 40
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def read_chamfer(metrics_path: str):
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path) as f:
        for line in f:
            if line.startswith("chamfer_distance"):
                try:
                    return float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return None


def collect_tag(tag_dir: str, n_min: int, n_max: int):
    """
    Scan all sequence sub-dirs inside tag_dir and collect Chamfer distances.

    Returns:
        ns    : sorted list of n_frames with at least one result
        means : np.array of mean CD per n_frames
        stds  : np.array of std  CD per n_frames
        counts: number of sequences contributing to each point
    """
    if not os.path.isdir(tag_dir):
        return [], np.array([]), np.array([]), []

    seq_dirs = [
        os.path.join(tag_dir, d)
        for d in sorted(os.listdir(tag_dir))
        if os.path.isdir(os.path.join(tag_dir, d))
    ]

    cd_by_n: dict[int, list[float]] = {}
    for seq_dir in seq_dirs:
        for n in range(n_min, n_max + 1):
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


# ── plot ──────────────────────────────────────────────────────────────────────

MASK_CONFIGS = [
    ("mask_25pct", "25%",  "#1565C0", "-",  "o"),
    ("mask_50pct", "50%",  "#C62828", "--", "s"),
]


def make_plot(series: dict, n_min: int, n_max: int, out_path: str):
    """series : {tag: (ns, means, stds, counts)}"""
    if not series:
        print("No data found – skipping plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for tag, label, color, ls, mk in MASK_CONFIGS:
        if tag not in series:
            continue
        ns, means, stds, counts = series[tag]
        ns_arr = np.array(ns)
        n_seq  = counts[0] if counts else 0

        ax.plot(ns_arr, means,
                marker=mk, color=color, linestyle=ls,
                linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgewidth=2,
                label=f"mask_ratio = {label}  (avg over {n_seq} sequences)",
                zorder=3)
        ax.fill_between(ns_arr, means - stds, means + stds,
                        color=color, alpha=0.12, zorder=2)

        # annotate the final (rightmost) value
        ax.annotate(f"{means[-1]:.3f}",
                    xy=(ns_arr[-1], means[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=color, fontweight="bold", va="center")

    ax.set_title("Random Masking — Chamfer Distance vs Number of Frames",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Number of input frames (all masked)", fontsize=12, labelpad=8)
    ax.set_ylabel("Chamfer Distance  (↓ lower is better)", fontsize=12, labelpad=8)

    all_ns = sorted({n for (ns, *_) in series.values() for n in ns})
    ax.set_xticks(all_ns)
    ax.tick_params(axis="both", labelsize=10)

    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC",
              loc="upper right", handlelength=2.5)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.margins(y=0.08)

    fig.suptitle(
        "DUSt3R — Averaged over 6 CO3D sequences\n"
        "(teddybear, hydrant, cup, bottle, toybus, toytrain)\n"
        "Mean ± 1 std  |  All frames masked (random patches)",
        fontsize=11, color="#444444", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r/mask_multi")
    parser.add_argument("--n_min",  type=int, default=2)
    parser.add_argument("--n_max",  type=int, default=20)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    root    = args.results_root
    out_dir = args.output_dir or root
    os.makedirs(out_dir, exist_ok=True)

    series = {}
    for tag, label, *_ in MASK_CONFIGS:
        tag_dir = os.path.join(root, tag)
        ns, means, stds, counts = collect_tag(tag_dir, args.n_min, args.n_max)
        if ns:
            series[tag] = (ns, means, stds, counts)
            print(f"{tag}: {counts[0] if counts else 0} sequences, "
                  f"{len(ns)} n_frames points, "
                  f"CD range [{means.min():.4f}, {means.max():.4f}]")
        else:
            print(f"{tag}: no results yet")

    make_plot(series, args.n_min, args.n_max,
              os.path.join(out_dir, "masking_compare.png"))

    # text summary
    if series:
        tags = [t for t, *_ in MASK_CONFIGS if t in series]
        all_n = sorted({n for (ns, *_) in series.values() for n in ns})
        maps  = {t: dict(zip(ns, means)) for t, (ns, means, *_) in series.items()}
        print(f"\n  Chamfer Distance (mean across 6 sequences)")
        print(f"  {'n':>4}", end="")
        for t in tags:
            print(f"   {t:>12}", end="")
        print()
        for n in all_n:
            print(f"  {n:>4}", end="")
            for t in tags:
                val = maps[t].get(n, float("nan"))
                print(f"  {val:>12.4f}", end="")
            print()


if __name__ == "__main__":
    main()
