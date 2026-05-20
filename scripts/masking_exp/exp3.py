"""
Experiment 3 (new): n_frames sweep from 2 to 40, ALL frames masked,
mask_ratio=0.25, num_patches=3.

Metrics: Chamfer Distance, Hausdorff Distance, F1 Score, PSNR d1
Comparison: plots Exp3 (mask=25%) vs Exp4 (mask=50%) for all 4 metrics.

Usage:
    python scripts/exp3.py                          # run inference + plot
    python scripts/exp3.py --plot_only              # just re-plot
    python scripts/exp3.py --compare_only           # only generate comparison plot
"""

import os
import sys
import argparse
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEQ_DIR    = "data/co3d/teddybear/101_11758_21048"
MASKED_DIR = "outputs/masked_frames/teddybear/101_11758_21048/mask_25pct"
OUTPUT_DIR = "outputs/dust3r/exp3"
EXP4_DIR   = "outputs/dust3r/exp4"
MASK_RATIO = 0.25
INFERENCE_SCRIPT = "scripts/run_dust3r_inference.py"

METRICS = [
    ("chamfer_distance", "Chamfer Distance",   "lower is better"),
    ("hausdorff",        "Hausdorff Distance", "lower is better"),
    ("f1",               "F1 Score",           "higher is better"),
    ("psnr_d1",          "PSNR d1 (dB)",       "higher is better"),
]


def read_metrics(path: str) -> dict:
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                k, v = line.split(':', 1)
                try:
                    data[k.strip()] = float(v.strip().replace(',', ''))
                except ValueError:
                    pass
    return data


def run_one(n_frames: int, seq_dir: str, masked_dir: str,
            output_dir: str, mask_ratio: float,
            dust3r_dir: str = "dust3r") -> str:
    out_subdir = os.path.join(output_dir, f"frames_{n_frames:02d}")
    os.makedirs(out_subdir, exist_ok=True)
    cmd = [
        sys.executable, INFERENCE_SCRIPT,
        "--sequence_dir", seq_dir,
        "--dust3r_dir",   dust3r_dir,
        "--n_frames",     str(n_frames),
        "--n_masked",     str(n_frames),
        "--masked_dir",   masked_dir,
        "--output_dir",   out_subdir,
        "--mask_ratio",   str(mask_ratio),
    ]
    print(f"\n>>> n_frames={n_frames} ...")
    subprocess.run(cmd, check=True)
    return out_subdir


def collect(output_dir: str, n_min: int, n_max: int) -> list:
    results = []
    for n in range(n_min, n_max + 1):
        p = os.path.join(output_dir, f"frames_{n:02d}", "metrics.txt")
        if not os.path.isfile(p):
            continue
        m = read_metrics(p)
        m['n_frames'] = n
        results.append(m)
    results.sort(key=lambda x: x['n_frames'])
    return results


def plot_single(results: list, output_dir: str, mask_label: str, color: str):
    """4-panel line plot for one experiment."""
    if not results:
        return
    n_frames = [r['n_frames'] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (key, label, note) in zip(axes, METRICS):
        vals = [r.get(key, float('nan')) for r in results]
        ax.plot(n_frames, vals, marker='o', linewidth=2, markersize=6,
                color=color, markerfacecolor='white', markeredgewidth=2)
        for x, y in zip(n_frames, vals):
            ax.annotate(f'{y:.3f}', (x, y),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=7)
        ax.set_xlabel('Number of Frames (all masked)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}  ({note})', fontsize=11, fontweight='bold')
        ax.set_xticks(n_frames)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'DUSt3R Reconstruction Quality vs. Frames — {mask_label}\n'
        f'Teddybear · 101_11758_21048',
        fontsize=13
    )
    plt.tight_layout()
    out = os.path.join(output_dir, 'chamfer_vs_n_frames.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Single-exp plot saved to: {out}")


def plot_comparison(exp3: list, exp4: list, output_dir: str):
    """4-panel comparison: Exp3 (25%) vs Exp4 (50%)."""
    if not exp3 and not exp4:
        print("No data for comparison plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    colors = {'exp3': '#2196F3', 'exp4': '#E91E63'}

    for ax, (key, label, note) in zip(axes, METRICS):
        if exp3:
            n3 = [r['n_frames'] for r in exp3]
            v3 = [r.get(key, float('nan')) for r in exp3]
            ax.plot(n3, v3, marker='o', linewidth=2, markersize=6,
                    color=colors['exp3'], markerfacecolor='white',
                    markeredgewidth=2, label='Exp3: mask=25%, patches=3')
        if exp4:
            n4 = [r['n_frames'] for r in exp4]
            v4 = [r.get(key, float('nan')) for r in exp4]
            ax.plot(n4, v4, marker='^', linewidth=2, markersize=6,
                    color=colors['exp4'], markerfacecolor='white',
                    markeredgewidth=2, label='Exp4: mask=50%, patches=6')

        ax.set_xlabel('Number of Frames (all masked)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}  ({note})', fontsize=11, fontweight='bold')
        all_n = sorted(set(
            ([r['n_frames'] for r in exp3] if exp3 else []) +
            ([r['n_frames'] for r in exp4] if exp4 else [])
        ))
        ax.set_xticks(all_n[::max(1, len(all_n)//20)])
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Exp3 vs Exp4 — Effect of Mask Ratio on 3D Reconstruction\n'
        'Teddybear · 101_11758_21048 · All Frames Masked',
        fontsize=13
    )
    plt.tight_layout()
    out = os.path.join(output_dir, 'exp3_vs_exp4.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {out}")


def print_table(exp3: list, exp4: list):
    keys = [k for k, _, _ in METRICS]
    header = f"{'n':>4}  " + "  ".join(f"{'Exp3_'+k[:6]:>12}  {'Exp4_'+k[:6]:>12}" for k in keys)
    print("\n" + header)
    print('-' * len(header))
    exp3_map = {r['n_frames']: r for r in exp3}
    exp4_map = {r['n_frames']: r for r in exp4}
    all_n = sorted(set(list(exp3_map) + list(exp4_map)))
    for n in all_n:
        row = f"{n:>4}  "
        for k in keys:
            v3 = f"{exp3_map[n][k]:.5f}" if n in exp3_map and k in exp3_map[n] else "  —  "
            v4 = f"{exp4_map[n][k]:.5f}" if n in exp4_map and k in exp4_map[n] else "  —  "
            row += f"  {v3:>12}  {v4:>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence_dir',  default=SEQ_DIR)
    parser.add_argument('--masked_dir',    default=MASKED_DIR)
    parser.add_argument('--output_dir',    default=OUTPUT_DIR)
    parser.add_argument('--exp4_dir',      default=EXP4_DIR)
    parser.add_argument('--n_frames_min',  type=int, default=2)
    parser.add_argument('--n_frames_max',  type=int, default=40)
    parser.add_argument('--mask_ratio',    type=float, default=MASK_RATIO)
    parser.add_argument('--plot_only',     action='store_true')
    parser.add_argument('--compare_only',  action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.plot_only and not args.compare_only:
        print("=" * 60)
        print("Experiment 3 (new): All-masked frames sweep")
        print(f"  n_frames : {args.n_frames_min} → {args.n_frames_max}")
        print(f"  mask_ratio={args.mask_ratio}, ALL frames masked")
        print("=" * 60)
        for n in range(args.n_frames_min, args.n_frames_max + 1):
            metrics_path = os.path.join(args.output_dir, f"frames_{n:02d}", "metrics.txt")
            if os.path.isfile(metrics_path):
                print(f"  [skip] n={n} already done")
                continue
            run_one(n, args.sequence_dir, args.masked_dir,
                    args.output_dir, args.mask_ratio)

    exp3 = collect(args.output_dir, args.n_frames_min, args.n_frames_max)
    exp4 = collect(args.exp4_dir,   args.n_frames_min, args.n_frames_max)

    if not exp3:
        print("No Exp3 results found.")
        return

    print_table(exp3, exp4)
    plot_single(exp3, args.output_dir, 'Exp3 (mask=25%, patches=3)', '#2196F3')
    plot_comparison(exp3, exp4, args.output_dir)


if __name__ == '__main__':
    main()
