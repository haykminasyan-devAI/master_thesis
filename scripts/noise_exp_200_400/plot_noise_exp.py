"""
Line plot: Number of frames vs Chamfer Distance for Gaussian noise std=200 and std=400,
averaged over 6 CO3D categories (teddybear, hydrant, cup, bottle, toybus, toytrain).

Expected directory structure:
    outputs/dust3r/noise_exp_200_400/
        noise_s200/
            teddybear_101_11758_21048/frames_02/metrics.txt
            ...
        noise_s400/
            ...

Usage:
    python scripts/noise_exp_200_400/plot_noise_exp.py \\
        --results_root outputs/dust3r/noise_exp_200_400
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NOISE_STDS  = [200, 400]
COLORS      = ["#1565C0", "#B71C1C"]   # blue, red
MARKERS     = ["o", "s"]
LINESTYLES  = ["-", "--"]


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

    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor("#F7F9FC")

    ax     = fig.add_axes([0.07, 0.12, 0.56, 0.72])
    ax_bar = fig.add_axes([0.70, 0.12, 0.27, 0.72])
    for a in [ax, ax_bar]:
        a.set_facecolor("#F7F9FC")

    all_means_flat = []
    results = {}
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
        results[std] = (ns, means, stds_v, n_seq)
        all_means_flat.extend(means.tolist())

        ax.plot(ns, means,
                color=COLORS[i], linestyle=LINESTYLES[i], marker=MARKERS[i],
                linewidth=2.5, markersize=8,
                markerfacecolor="white", markeredgewidth=2.2,
                label=f"Noise σ={std}  (n_seq={n_seq})",
                zorder=3)
        ax.fill_between(ns, means - stds_v, means + stds_v,
                        color=COLORS[i], alpha=0.13, zorder=2)

        offsets = [10, -14]
        ax.annotate(f"{means[-1]:.3f}",
                    xy=(ns[-1], means[-1]),
                    xytext=(8, offsets[i]), textcoords="offset points",
                    fontsize=9, color=COLORS[i], fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=COLORS[i], lw=0.8))

    if all_means_flat:
        ax.set_ylim(min(all_means_flat) * 0.97, max(all_means_flat) * 1.03)

    all_ns = sorted({n for std in NOISE_STDS for n in data.get(std, {})})
    ax.set_xticks(all_ns)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("Number of input frames", fontsize=12, labelpad=6)
    ax.set_ylabel("Chamfer Distance  (↓ lower is better)", fontsize=12, labelpad=6)
    ax.set_title("Chamfer Distance vs Number of Frames\n(Gaussian Noise σ=200 vs σ=400)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=11, framealpha=0.92, edgecolor="#CCCCCC",
              loc="upper right", handlelength=2.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Bar chart: final values at n_max
    bar_labels, bar_vals, bar_errs, bar_colors = [], [], [], []
    for i, std in enumerate(NOISE_STDS):
        if std not in results:
            continue
        ns, means, stds_v, _ = results[std]
        bar_labels.append(f"σ={std}")
        bar_vals.append(means[-1])
        bar_errs.append(stds_v[-1])
        bar_colors.append(COLORS[i])

    x_pos = np.arange(len(bar_labels))
    bars = ax_bar.bar(x_pos, bar_vals, yerr=bar_errs, color=bar_colors,
                      width=0.5, capsize=6, edgecolor="white", linewidth=1.2,
                      error_kw=dict(elinewidth=1.5, ecolor="#333333"), zorder=3)
    for bar, val in zip(bars, bar_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, fontsize=12)
    ax_bar.set_ylabel("Chamfer Distance", fontsize=11, labelpad=6)
    n_max = max(all_ns) if all_ns else 20
    ax_bar.set_title(f"Final values  (n={n_max})", fontsize=12, fontweight="bold", pad=10)
    if bar_vals:
        ax_bar.set_ylim(min(bar_vals) * 0.97, max(bar_vals) * 1.05)
    ax_bar.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_bar.set_axisbelow(True)

    fig.suptitle(
        f"DUSt3R Gaussian Noise Experiment  |  "
        f"Averaged over {n_seq_max} CO3D sequences  "
        f"(teddybear, hydrant, cup, bottle, toybus, toytrain)  |  Shaded = ±1 std",
        fontsize=9, color="#555555", y=0.98)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r/noise_exp_200_400")
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
