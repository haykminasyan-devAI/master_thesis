"""
Line plot: Number of frames vs Chamfer Distance for different viewpoint angle ranges.

Expected directory structure:
    outputs/dust3r/angle_range_exp/
        range_0_60deg/
            teddybear_101_11758_21048/frames_02/metrics.txt
            ...
        range_0_90deg/
            ...
        range_0_180deg/
            ...

Usage:
    python scripts/angle_range_exp/plot_angle_range_exp.py \
        --results_root outputs/dust3r/angle_range_exp \
        --n_min 2 --n_max 10
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RANGE_TAGS   = ["range_0_60deg", "range_0_90deg", "range_0_180deg"]
RANGE_LABELS = ["0°–60°  (narrow)", "0°–90°  (quarter)", "0°–180°  (half)"]
COLORS       = ["#E65100", "#1565C0", "#2E7D32"]
MARKERS      = ["o", "s", "^"]
LINESTYLES   = ["-", "--", "-."]


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


def collect(results_root: str, n_min: int, n_max: int):
    data = {tag: {} for tag in RANGE_TAGS}

    for tag in RANGE_TAGS:
        tag_dir = os.path.join(results_root, tag)
        if not os.path.isdir(tag_dir):
            print(f"  [missing] {tag_dir}")
            continue

        for seq_dir in sorted(os.listdir(tag_dir)):
            seq_path = os.path.join(tag_dir, seq_dir)
            if not os.path.isdir(seq_path):
                continue

            for n in range(n_min, n_max + 1):
                metrics = os.path.join(seq_path, f"frames_{n:02d}", "metrics.txt")
                cd = read_chamfer(metrics)
                if cd is not None:
                    data[tag].setdefault(n, []).append(cd)

    return data


def plot(data, n_min, n_max, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(11, 6))
    fig.patch.set_facecolor("#F7F9FC")
    ax = fig.add_axes([0.09, 0.12, 0.86, 0.70])
    ax.set_facecolor("#F7F9FC")

    all_means_flat = []
    n_seq_max = 0

    for i, tag in enumerate(RANGE_TAGS):
        ns_dict = data.get(tag, {})
        if not ns_dict:
            print(f"  No data for {tag}")
            continue

        ns     = sorted(ns_dict.keys())
        means  = np.array([np.mean(ns_dict[n]) for n in ns])
        stds_v = np.array([np.std(ns_dict[n])  for n in ns])
        n_seq  = max(len(ns_dict[n]) for n in ns)
        n_seq_max = max(n_seq_max, n_seq)
        all_means_flat.extend(means.tolist())

        ax.plot(ns, means,
                color=COLORS[i], linestyle=LINESTYLES[i], marker=MARKERS[i],
                linewidth=2.5, markersize=9,
                markerfacecolor="white", markeredgewidth=2.2,
                label=f"{RANGE_LABELS[i]}  (n_seq={n_seq})",
                zorder=3)
        ax.fill_between(ns, means - stds_v, means + stds_v,
                        color=COLORS[i], alpha=0.13, zorder=2)

        offsets = [10, 0, -10]
        ax.annotate(f"{means[-1]:.3f}",
                    xy=(ns[-1], means[-1]),
                    xytext=(8, offsets[i]), textcoords="offset points",
                    fontsize=9, color=COLORS[i], fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=COLORS[i], lw=0.8))

    if all_means_flat:
        y_min = min(all_means_flat) * 0.97
        y_max = max(all_means_flat) * 1.03
        ax.set_ylim(y_min, y_max)

    all_ns = sorted({n for tag in RANGE_TAGS for n in data.get(tag, {})})
    ax.set_xticks(all_ns)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("Number of input frames", fontsize=12, labelpad=6)
    ax.set_ylabel("Chamfer Distance  (↓ lower is better)", fontsize=12, labelpad=6)
    ax.set_title("Chamfer Distance vs Number of Frames\n(Viewpoint Angle Range Experiment, Clean Frames)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10, framealpha=0.92, edgecolor="#CCCCCC",
              loc="upper right", handlelength=2.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"DUSt3R Angle Range Experiment  |  "
        f"Averaged over {n_seq_max} CO3D sequences  "
        f"(teddybear, hydrant, cup, bottle, toybus, toytrain)  |  "
        f"Shaded = ±1 std",
        fontsize=9, color="#555555", y=0.98)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r/angle_range_exp")
    parser.add_argument("--n_min", type=int, default=2)
    parser.add_argument("--n_max", type=int, default=10)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_root
    os.makedirs(out_dir, exist_ok=True)

    print(f"Collecting from: {args.results_root}  (n={args.n_min}..{args.n_max})\n")
    data = collect(args.results_root, args.n_min, args.n_max)

    print("\nSummary:")
    for tag in RANGE_TAGS:
        for n in sorted(data.get(tag, {})):
            vals = data[tag][n]
            print(f"  {tag}  n={n:>2}  mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  ({len(vals)} seqs)")

    out_path = os.path.join(out_dir, "angle_range_compare.png")
    plot(data, args.n_min, args.n_max, out_path)


if __name__ == "__main__":
    main()
