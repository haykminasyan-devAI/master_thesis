"""
Average Chamfer Distance across 10 CO3D sequences and plot blur vs noise comparisons.

Directory structure expected:
    outputs/dust3r/degrade_multi/
        blur_s1/
            teddybear_101_11758_21048/frames_02/metrics.txt
            teddybear_101_11758_21048/frames_03/metrics.txt
            ...
            hydrant_106_12648_23157/frames_02/metrics.txt
            ...
        blur_s3/  blur_s5/
        noise_s10/ noise_s25/ noise_s50/

For each degradation tag and each n_frames:
  - collects Chamfer distance from every sequence that has results
  - computes mean ± std across sequences
  - plots mean curve with shaded ±1 std band

Produces two separate figures:
  blur_compare.png  — blur  σ = 1, 3, 5
  noise_compare.png — noise σ = 10, 25, 50

Usage:
    python scripts/gaussian_noise_and_blur_exps/plot_degrade_multi_seq.py \\
        --results_root outputs/dust3r/degrade_multi \\
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
        ns   : sorted list of n_frames values that have at least one result
        means: np.array of mean CD per n_frames
        stds : np.array of std  CD per n_frames
        counts: number of sequences contributing to each point
    """
    if not os.path.isdir(tag_dir):
        return [], np.array([]), np.array([]), []

    seq_dirs = [
        os.path.join(tag_dir, d)
        for d in sorted(os.listdir(tag_dir))
        if os.path.isdir(os.path.join(tag_dir, d))
    ]

    # cd_by_n[n] = list of CD values from different sequences
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


# ── style ─────────────────────────────────────────────────────────────────────

BLUR_COLORS   = ["#64B5F6", "#1565C0", "#0D2A6B"]   # light → dark blue
NOISE_COLORS  = ["#EF9A9A", "#C62828", "#4A0000"]    # light → dark red
LINESTYLES    = ["-", "--", ":"]
MARKERS       = ["o", "s", "^"]


def _make_plot(series: dict, colors: list, title: str, suptitle: str,
               param_label: str, n_min: int, n_max: int, out_path: str):
    """
    series  : {param_value: (ns, means, stds, counts)}
    colors  : one colour per param_value (sorted)
    """
    if not series:
        print(f"No data for {out_path} – skipping.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    sorted_items = sorted(series.items())
    for i, (param, (ns, means, stds, counts)) in enumerate(sorted_items):
        ns_arr = np.array(ns)
        color  = colors[i % len(colors)]
        ls     = LINESTYLES[i % len(LINESTYLES)]
        mk     = MARKERS[i % len(MARKERS)]
        n_seq  = counts[0] if counts else 0

        ax.plot(ns_arr, means,
                marker=mk, color=color, linestyle=ls,
                linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgewidth=2,
                label=f"{param_label} = {param}  (avg over {n_seq} sequences)",
                zorder=3)
        ax.fill_between(ns_arr, means - stds, means + stds,
                        color=color, alpha=0.12, zorder=2)

        # annotate the final (rightmost) value
        ax.annotate(f"{means[-1]:.3f}",
                    xy=(ns_arr[-1], means[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=color, fontweight="bold", va="center")

    # axis labels & title
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Number of input frames", fontsize=12, labelpad=8)
    ax.set_ylabel("Chamfer Distance  (↓ lower is better)", fontsize=12, labelpad=8)

    # x-ticks at every frame value
    all_ns = sorted({n for (_, (ns, *_)) in series.items() for n in ns})
    ax.set_xticks(all_ns)
    ax.tick_params(axis="both", labelsize=10)

    # legend
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC",
              loc="upper right", handlelength=2.5)

    # subtle grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    # y-axis margin
    ax.margins(y=0.08)

    fig.suptitle(suptitle, fontsize=11, color="#444444", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r/degrade_multi",
                        help="Root dir containing blur_sX / noise_sX subdirectories")
    parser.add_argument("--n_min",  type=int, default=2)
    parser.add_argument("--n_max",  type=int, default=40)
    parser.add_argument("--output_dir", default=None,
                        help="Where to save plots (default: results_root)")
    args = parser.parse_args()

    root    = args.results_root
    out_dir = args.output_dir or root
    os.makedirs(out_dir, exist_ok=True)

    # ── collect blur results ──────────────────────────────────────────────────
    blur_series = {}
    for sigma in [1, 3, 5]:
        tag_dir = os.path.join(root, f"blur_s{sigma}")
        ns, means, stds, counts = collect_tag(tag_dir, args.n_min, args.n_max)
        if ns:
            blur_series[sigma] = (ns, means, stds, counts)
            print(f"blur_s{sigma}: {counts[0] if counts else 0} sequences, "
                  f"{len(ns)} n_frames points")

    # ── collect noise results ─────────────────────────────────────────────────
    noise_series = {}
    for std in [10, 25, 50]:
        tag_dir = os.path.join(root, f"noise_s{std}")
        ns, means, stds, counts = collect_tag(tag_dir, args.n_min, args.n_max)
        if ns:
            noise_series[std] = (ns, means, stds, counts)
            print(f"noise_s{std}: {counts[0] if counts else 0} sequences, "
                  f"{len(ns)} n_frames points")

    if not blur_series and not noise_series:
        print("No results found yet – nothing to plot.")
        return

    base_suptitle = ("DUSt3R — Averaged over 6 CO3D sequences\n"
                     "(teddybear, hydrant, cup, bottle, toybus, toytrain)\n"
                     "Mean ± 1 std  |  All frames degraded")

    # ── blur plot ─────────────────────────────────────────────────────────────
    _make_plot(
        series      = blur_series,
        colors      = BLUR_COLORS,
        title       = "Gaussian Blur — Chamfer Distance vs Number of Frames",
        suptitle    = base_suptitle,
        param_label = "σ (pixels)",
        n_min       = args.n_min,
        n_max       = args.n_max,
        out_path    = os.path.join(out_dir, "blur_compare.png"),
    )

    # ── noise plot ────────────────────────────────────────────────────────────
    _make_plot(
        series      = noise_series,
        colors      = NOISE_COLORS,
        title       = "Gaussian Noise (μ=0) — Chamfer Distance vs Number of Frames",
        suptitle    = base_suptitle,
        param_label = "σ (0–255 scale)",
        n_min       = args.n_min,
        n_max       = args.n_max,
        out_path    = os.path.join(out_dir, "noise_compare.png"),
    )

    # ── text summary ──────────────────────────────────────────────────────────
    for name, series in [("BLUR", blur_series), ("NOISE", noise_series)]:
        if not series:
            continue
        params = sorted(series.keys())
        print(f"\n  {name}  — mean Chamfer Distance (averaged across sequences)")
        print(f"  {'n':>4}", end="")
        for p in params:
            print(f"   {name.lower()[:3]}_s{p:>2}", end="")
        print()
        all_n = sorted({n for ns, *_ in series.values() for n in ns})
        maps  = {p: dict(zip(ns, means)) for p, (ns, means, *_) in series.items()}
        for n in all_n:
            print(f"  {n:>4}", end="")
            for p in params:
                val = maps[p].get(n, float("nan"))
                print(f"  {val:>9.4f}", end="")
            print()


if __name__ == "__main__":
    main()
