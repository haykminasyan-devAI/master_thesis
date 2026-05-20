"""
Line plot: Number of frames vs Chamfer Distance for Gaussian noise std=5, 10, 30, 50, 70,
averaged over 6 CO3D categories (teddybear, hydrant, cup, bottle, toybus, toytrain).

Expected directory structure:
    outputs/dust3r/noise_exp_30_50_70/
        noise_s5/
        noise_s10/
        noise_s30/
            teddybear_101_11758_21048/frames_02/metrics.txt
            ...
        noise_s50/
            ...
        noise_s70/
            ...

Usage:
    python scripts/noise_exp_30_50_70/plot_noise_exp.py \\
        --results_root outputs/dust3r/noise_exp_30_50_70
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NOISE_STDS  = [5, 10, 30, 50, 70]
COLORS      = ["#1565C0", "#00838F", "#1B5E20", "#E65100", "#6A1B9A"]  # blue, teal, green, orange, purple
MARKERS     = ["D", "v", "o", "s", "^"]
LINESTYLES  = [":", "-", "--", "-.", (0, (3, 1, 1, 1))]


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
    data = {std: {} for std in NOISE_STDS}
    for std in NOISE_STDS:
        tag_dir = os.path.join(results_root, f"noise_s{std}")
        if not os.path.isdir(tag_dir):
            print(f"  [missing] {tag_dir}")
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
    return data


def plot(data, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    all_means_flat = []
    n_seq_max = 0

    annotation_offsets = [10, -14, 10, -18, 14]

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
        all_means_flat.extend(means.tolist())

        ax.plot(ns, means,
                color=COLORS[i], linestyle=LINESTYLES[i], marker=MARKERS[i],
                linewidth=2.5, markersize=8,
                markerfacecolor="white", markeredgewidth=2.2,
                label=f"Noise σ={std}  (n_seq={n_seq})",
                zorder=3)
        ax.fill_between(ns, means - stds_v, means + stds_v,
                        color=COLORS[i], alpha=0.13, zorder=2)

        ax.annotate(f"{means[-1]:.3f}",
                    xy=(ns[-1], means[-1]),
                    xytext=(8, annotation_offsets[i]),
                    textcoords="offset points",
                    fontsize=9, color=COLORS[i], fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=COLORS[i], lw=0.8))

    if all_means_flat:
        ax.set_ylim(min(all_means_flat) * 0.97, max(all_means_flat) * 1.03)

    all_ns = sorted({n for std in NOISE_STDS for n in data.get(std, {})})
    ax.set_xticks(all_ns)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("Number of input frames", fontsize=12, labelpad=6)
    ax.set_ylabel("Chamfer Distance  (↓ lower is better)", fontsize=12, labelpad=6)
    ax.set_title("Chamfer Distance vs Number of Frames\n(Gaussian Noise σ=5, 10, 30, 50, 70)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=11, framealpha=0.92, edgecolor="#CCCCCC",
              loc="upper right", handlelength=2.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"DUSt3R Gaussian Noise Experiment  |  "
        f"Averaged over {n_seq_max} CO3D sequences  "
        f"(teddybear, hydrant, cup, bottle, toybus, toytrain)  |  Shaded = ±1 std",
        fontsize=9, color="#555555", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r/noise_exp_30_50_70")
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_root
    os.makedirs(out_dir, exist_ok=True)

    print(f"Collecting from: {args.results_root}\n")
    data = collect(args.results_root)

    print("\nSummary:")
    for std in NOISE_STDS:
        for n in sorted(data.get(std, {})):
            vals = data[std][n]
            print(f"  noise σ={std}  n={n:>2}  mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  ({len(vals)} sequences)")

    out_path = os.path.join(out_dir, "noise_exp_compare.png")
    plot(data, out_path)


if __name__ == "__main__":
    main()
