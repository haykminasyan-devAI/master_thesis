"""
Chamfer Distance comparison across Gaussian-blur and Gaussian-noise experiments.

Reads all   outputs/dust3r/frames_0_40_blur_sX/   and
            outputs/dust3r/frames_0_40_noise_sX/   directories
plus the existing mask-25% baseline (frames_0_40_0.25_3).

Produces TWO separate figures:
  blur_compare.png  – CD vs n_frames for blur  sigma = 1, 3, 5
  noise_compare.png – CD vs n_frames for noise std   = 10, 25, 50
A thin grey dashed line for the mask-25% baseline appears in both.

Usage (called automatically at the end of each SLURM job, or manually):
    python scripts/gaussian_noise_and_blur_exps/plot_degrade_compare.py \\
        --results_root outputs/dust3r \\
        --n_min 2 --n_max 40
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def read_chamfer(metrics_path: str) -> float | None:
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


def collect(exp_dir: str, n_min: int, n_max: int) -> tuple[list[int], list[float]]:
    """Return (n_values, cd_values) for all completed frame counts."""
    ns, cds = [], []
    for n in range(n_min, n_max + 1):
        p = os.path.join(exp_dir, f"frames_{n:02d}", "metrics.txt")
        cd = read_chamfer(p)
        if cd is not None:
            ns.append(n)
            cds.append(cd)
    return ns, cds


# ── palette (colour-blind-friendly) ──────────────────────────────────────────

BLUR_COLORS  = ["#90CAF9", "#1E88E5", "#0D47A1"]   # light → dark blue
NOISE_COLORS = ["#FFAB91", "#E53935", "#7B1010"]    # light → dark red
MASK_COLOR   = "#9E9E9E"                             # grey baseline


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_root", default="outputs/dust3r",
                        help="Root directory that contains all experiment sub-dirs")
    parser.add_argument("--n_min", type=int, default=2)
    parser.add_argument("--n_max", type=int, default=40)
    parser.add_argument("--output_dir", default=None,
                        help="Where to save the plot (default: results_root)")
    args = parser.parse_args()

    root       = args.results_root
    out_dir    = args.output_dir or root
    os.makedirs(out_dir, exist_ok=True)

    # ── collect all series ────────────────────────────────────────────────────

    blur_series = {}   # sigma → (ns, cds)
    for sigma in [1, 3, 5]:
        d = os.path.join(root, f"frames_0_40_blur_s{sigma}")
        ns, cds = collect(d, args.n_min, args.n_max)
        if ns:
            blur_series[sigma] = (ns, cds)

    noise_series = {}  # std → (ns, cds)
    for std in [10, 25, 50]:
        d = os.path.join(root, f"frames_0_40_noise_s{std}")
        ns, cds = collect(d, args.n_min, args.n_max)
        if ns:
            noise_series[std] = (ns, cds)

    # mask-25% baseline (already computed)
    mask_ns, mask_cds = collect(
        os.path.join(root, "frames_0_40_0.25_3"), args.n_min, args.n_max
    )

    if not blur_series and not noise_series:
        print("No results found yet – nothing to plot.")
        return

    marker_kw = dict(linewidth=2, markersize=5, markerfacecolor="white",
                     markeredgewidth=1.5)

    def _add_baseline(ax):
        if mask_ns:
            ax.plot(mask_ns, mask_cds, color=MASK_COLOR, linewidth=1.5,
                    linestyle="--", label="Mask 25 %  (baseline)", zorder=1)

    def _style_ax(ax, title, n_min, n_max):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of frames (all degraded)", fontsize=11)
        ax.set_ylabel("Chamfer Distance  (↓ better)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(n_min, n_max + 1, 2))
        ax.tick_params(axis="x", rotation=45)

    # ── figure 1: blur ────────────────────────────────────────────────────────
    if blur_series:
        fig, ax = plt.subplots(figsize=(10, 6))
        _add_baseline(ax)
        for (sigma, (ns, cds)), color in zip(sorted(blur_series.items()), BLUR_COLORS):
            ax.plot(ns, cds, marker="o", color=color,
                    label=f"Blur  σ={sigma} px", **marker_kw)
        _style_ax(ax, "Gaussian Blur — Chamfer Distance vs # Frames", args.n_min, args.n_max)
        fig.suptitle(
            "DUSt3R — Gaussian Blur Degradation\n"
            "Teddybear · 101_11758_21048 · All frames blurred",
            fontsize=12,
        )
        plt.tight_layout()
        out_blur = os.path.join(out_dir, "blur_compare.png")
        plt.savefig(out_blur, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Blur plot saved  → {out_blur}")

    # ── figure 2: noise ───────────────────────────────────────────────────────
    if noise_series:
        fig, ax = plt.subplots(figsize=(10, 6))
        _add_baseline(ax)
        for (std, (ns, cds)), color in zip(sorted(noise_series.items()), NOISE_COLORS):
            ax.plot(ns, cds, marker="s", color=color,
                    label=f"Noise  σ={std}  (0–255)", **marker_kw)
        _style_ax(ax, "Gaussian Noise (μ=0) — Chamfer Distance vs # Frames", args.n_min, args.n_max)
        fig.suptitle(
            "DUSt3R — Gaussian Noise Degradation\n"
            "Teddybear · 101_11758_21048 · All frames noisy",
            fontsize=12,
        )
        plt.tight_layout()
        out_noise = os.path.join(out_dir, "noise_compare.png")
        plt.savefig(out_noise, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Noise plot saved → {out_noise}")

    # ── text summary ──────────────────────────────────────────────────────────
    print("\n=== Chamfer Distance summary ===")
    if blur_series:
        print("\n  BLUR (all frames, sigma px):")
        print(f"  {'n':>4}", end="")
        for s in sorted(blur_series):
            print(f"   blur_s{s:>2}", end="")
        print()
        all_n = sorted({n for ns, _ in blur_series.values() for n in ns})
        maps  = {s: dict(zip(*v)) for s, v in blur_series.items()}
        for n in all_n:
            print(f"  {n:>4}", end="")
            for s in sorted(blur_series):
                val = maps[s].get(n, float("nan"))
                print(f"  {val:>9.4f}", end="")
            print()

    if noise_series:
        print("\n  NOISE (all frames, std 0-255):")
        print(f"  {'n':>4}", end="")
        for s in sorted(noise_series):
            print(f"  noise_{s:>3}", end="")
        print()
        all_n = sorted({n for ns, _ in noise_series.values() for n in ns})
        maps  = {s: dict(zip(*v)) for s, v in noise_series.items()}
        for n in all_n:
            print(f"  {n:>4}", end="")
            for s in sorted(noise_series):
                val = maps[s].get(n, float("nan"))
                print(f"  {val:>9.4f}", end="")
            print()


if __name__ == "__main__":
    main()
