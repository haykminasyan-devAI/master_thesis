"""
Line plot: Number of frames vs Chamfer Distance for Gaussian noise std=200 and std=300,
averaged over 6 CO3D categories (teddybear, hydrant, cup, bottle, toybus, toytrain).

Expected directory structure:
    outputs/dust3r/noise_heavy/
        noise_s200/
            teddybear_101_11758_21048/frames_10/metrics.txt
            teddybear_101_11758_21048/frames_14/metrics.txt
            ...
        noise_s300/
            ...

Usage:
    python scripts/noise_heavy_exp/plot_noise_heavy.py \\
        --results_root outputs/dust3r/noise_heavy
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NOISE_STDS = [200, 300]
COLORS     = ["#E57373", "#B71C1C"]   # light red, dark red
MARKERS    = ["o", "s"]
LINESTYLES = ["-", "--"]


def read_chamfer(path: str):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if line.startswith("chamfer_distance"):
                try:
                    return float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return None


def collect(results_root: str):
    """
    Returns:
        data[std][n] = [cd_seq1, cd_seq2, ...]
    """
    data = {std: {} for std in NOISE_STDS}

    for std in NOISE_STDS:
        tag_dir = os.path.join(results_root, f"noise_s{std}")
        if not os.path.isdir(tag_dir):
            print(f"  [missing dir] {tag_dir}")
            continue

        for seq_dir in sorted(os.listdir(tag_dir)):
            seq_path = os.path.join(tag_dir, seq_dir)
            if not os.path.isdir(seq_path):
                continue

            for frames_dir in sorted(os.listdir(seq_path)):
                if not frames_dir.startswith("frames_"):
                    continue
                try:
                    n = int(frames_dir.split("_")[1])
                except ValueError:
                    continue

                metrics = os.path.join(seq_path, frames_dir, "metrics.txt")
                cd = read_chamfer(metrics)
                if cd is not None:
                    data[std].setdefault(n, []).append(cd)
                    print(f"  noise_s{std}  n={n:>2}  {seq_dir}  CD={cd:.4f}")
                else:
                    print(f"  [missing] {metrics}")

    return data


def plot(data, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    has_data = False
    n_seq_max = 0
    for i, std in enumerate(NOISE_STDS):
        ns_dict = data.get(std, {})
        if not ns_dict:
            print(f"  No data for noise_s{std}")
            continue

        ns     = sorted(ns_dict.keys())
        means  = np.array([np.mean(ns_dict[n]) for n in ns])
        stds_v = np.array([np.std(ns_dict[n])  for n in ns])
        n_seq  = max(len(ns_dict[n]) for n in ns)
        n_seq_max = max(n_seq_max, n_seq)

        ax.plot(ns, means,
                color=COLORS[i], linestyle=LINESTYLES[i], marker=MARKERS[i],
                linewidth=2.5, markersize=8,
                markerfacecolor="white", markeredgewidth=2,
                label=f"Noise σ={std}  (avg over {n_seq} sequences)",
                zorder=3)
        ax.fill_between(ns, means - stds_v, means + stds_v,
                        color=COLORS[i], alpha=0.15, zorder=2)

        # annotate last point
        ax.annotate(f"{means[-1]:.4f}",
                    xy=(ns[-1], means[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=COLORS[i], fontweight="bold", va="center")

        has_data = True

    if not has_data:
        print("No data to plot.")
        plt.close()
        return

    ax.set_xlabel("Number of input frames", fontsize=13, labelpad=8)
    ax.set_ylabel("Chamfer Distance  (↓ lower is better)", fontsize=13, labelpad=8)
    ax.set_title("Gaussian Noise — Chamfer Distance vs Number of Frames",
                 fontsize=15, fontweight="bold", pad=12)

    # x-ticks at every evaluated n
    all_ns = sorted({n for std in NOISE_STDS for n in data.get(std, {})})
    ax.set_xticks(all_ns)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(bottom=0)

    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC",
              loc="upper right", handlelength=2.5)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.margins(y=0.08)

    fig.suptitle(
        f"Averaged over {n_seq_max} CO3D sequences  "
        f"(teddybear, hydrant, cup, bottle, toybus, toytrain)\n"
        f"Shaded band = ±1 std across sequences",
        fontsize=9, color="#555555", y=1.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r/noise_heavy")
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_root
    os.makedirs(out_dir, exist_ok=True)

    print(f"Collecting results from: {args.results_root}\n")
    data = collect(args.results_root)

    print("\nSummary:")
    for std in NOISE_STDS:
        for n in sorted(data.get(std, {})):
            vals = data[std][n]
            print(f"  noise σ={std}  n={n:>2}  mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  ({len(vals)} sequences)")

    out_path = os.path.join(out_dir, "noise_heavy_compare.png")
    plot(data, out_path)


if __name__ == "__main__":
    main()
