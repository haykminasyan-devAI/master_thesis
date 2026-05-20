"""
Compute Earth Mover's Distance (EMD / Wasserstein-1) between predicted and GT
point clouds for noise_s200 and noise_s400 experiments, then plot EMD vs n_frames.

Both point clouds are subsampled to N_SUB points before EMD computation.

Usage:
    python scripts/noise_exp_200_400/plot_emd.py \
        --results_root outputs/dust3r/noise_exp_200_400 \
        --data_root    data/co3d
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
from plyfile import PlyData

NOISE_STDS  = [200, 400]
COLORS      = ["#1565C0", "#B71C1C"]
MARKERS     = ["o", "s"]
LINESTYLES  = ["-", "--"]
N_SUB       = 2000   # subsample size for EMD (larger = more accurate but slower)

SEQUENCES = {
    "teddybear": "101_11758_21048",
    "hydrant":   "106_12648_23157",
    "cup":       "12_100_593",
    "bottle":    "34_1397_4376",
    "toybus":    "111_13154_25988",
    "toytrain":  "104_12352_22039",
}


def load_ply_xyz(path: str, n_sub: int, rng) -> np.ndarray:
    """Load a PLY file and return subsampled xyz (n_sub, 3)."""
    data = PlyData.read(path)
    v = data["vertex"]
    xyz = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1).astype(np.float32)
    if len(xyz) > n_sub:
        idx = rng.choice(len(xyz), n_sub, replace=False)
        xyz = xyz[idx]
    elif len(xyz) < n_sub:
        # upsample by repeating
        idx = rng.choice(len(xyz), n_sub, replace=True)
        xyz = xyz[idx]
    return xyz


def compute_emd(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Compute Earth Mover's Distance between two equal-size point sets."""
    n = len(pred_xyz)
    # uniform weights
    a = np.ones(n) / n
    b = np.ones(n) / n
    # cost matrix: squared euclidean distances
    M = ot.dist(pred_xyz, gt_xyz, metric="euclidean")
    emd = ot.emd2(a, b, M)
    return float(emd)


def collect(results_root: str, data_root: str, n_frames_list: list):
    rng = np.random.default_rng(42)
    data = {std: {} for std in NOISE_STDS}

    for std in NOISE_STDS:
        tag_dir = os.path.join(results_root, f"noise_s{std}")
        if not os.path.isdir(tag_dir):
            print(f"  [missing] {tag_dir}")
            continue

        for cat, seq_id in SEQUENCES.items():
            seq_dir = os.path.join(tag_dir, f"{cat}_{seq_id}")
            if not os.path.isdir(seq_dir):
                print(f"  [missing seq] {seq_dir}")
                continue

            gt_ply = os.path.join(data_root, cat, seq_id, "pointcloud.ply")
            if not os.path.isfile(gt_ply):
                print(f"  [missing GT] {gt_ply}")
                continue

            gt_xyz = load_ply_xyz(gt_ply, N_SUB, rng)

            for n in n_frames_list:
                pred_ply = os.path.join(seq_dir, f"frames_{n:02d}", "predicted.ply")
                if not os.path.isfile(pred_ply):
                    print(f"  [missing pred] {pred_ply}")
                    continue

                pred_xyz = load_ply_xyz(pred_ply, N_SUB, rng)
                emd = compute_emd(pred_xyz, gt_xyz)
                data[std].setdefault(n, []).append(emd)
                print(f"  noise_s{std}  {cat}  n={n:>2}  EMD={emd:.4f}")

    return data


def plot(data, out_path, n_frames_list):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    all_means_flat = []
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

    ax.set_xticks(sorted({n for std in NOISE_STDS for n in data.get(std, {})}))
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("Number of input frames", fontsize=12, labelpad=6)
    ax.set_ylabel("Earth Mover's Distance  (↓ lower is better)", fontsize=12, labelpad=6)
    ax.set_title("Earth Mover's Distance vs Number of Frames\n(Gaussian Noise σ=200 vs σ=400)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=11, framealpha=0.92, edgecolor="#CCCCCC",
              loc="upper right", handlelength=2.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"DUSt3R Gaussian Noise Experiment  |  "
        f"Averaged over {n_seq_max} CO3D sequences  "
        f"(teddybear, hydrant, cup, bottle, toybus, toytrain)  |  "
        f"Point clouds subsampled to {N_SUB} pts  |  Shaded = ±1 std",
        fontsize=9, color="#555555", y=0.98)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r/noise_exp_200_400")
    parser.add_argument("--data_root",    default="data/co3d")
    parser.add_argument("--output_dir",   default=None)
    args = parser.parse_args()

    n_frames_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    out_dir = args.output_dir or args.results_root
    os.makedirs(out_dir, exist_ok=True)

    print(f"Computing EMD (subsample={N_SUB} pts per cloud)...\n")
    data = collect(args.results_root, args.data_root, n_frames_list)

    out_path = os.path.join(out_dir, "noise_exp_emd.png")
    plot(data, out_path, n_frames_list)


if __name__ == "__main__":
    main()
