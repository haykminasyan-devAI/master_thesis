"""
Experiment 2: How does reconstruction quality change as we increase
the total number of frames when ALL frames are masked?

Fixed:   mask_ratio=0.25, num_patches=3, ALL frames masked
Varying: n_frames from --n_frames_min to --n_frames_max
         Each run selects n_frames evenly-spaced frames from the sequence,
         ALL taken from the pre-masked directory.

Includes the n=10 all-masked result from Exp1 as a reference point.

Usage (run from project root inside dust3r conda env):
    python scripts/exp2.py
    python scripts/exp2.py --n_frames_min 11 --n_frames_max 20
"""

import os
import sys
import argparse
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── defaults ──────────────────────────────────────────────────────────────────
SEQ_DIR    = "data/co3d/teddybear/101_11758_21048"
MASKED_DIR = "outputs/masked_frames/teddybear/101_11758_21048/mask_25pct"
OUTPUT_DIR = "outputs/dust3r/exp2"
EXP1_SWEEP = "outputs/dust3r/sweep/masked_10of10/metrics.txt"   # all-masked baseline
MASK_RATIO = 0.25
INFERENCE_SCRIPT = "scripts/run_dust3r_inference.py"


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
    """Call run_dust3r_inference.py for one (n_frames, all-masked) run."""
    out_subdir = os.path.join(output_dir, f"frames_{n_frames:02d}")
    os.makedirs(out_subdir, exist_ok=True)

    cmd = [
        sys.executable, INFERENCE_SCRIPT,
        "--sequence_dir", seq_dir,
        "--dust3r_dir",   dust3r_dir,
        "--n_frames",     str(n_frames),
        "--n_masked",     str(n_frames),   # ALL frames masked
        "--masked_dir",   masked_dir,
        "--output_dir",   out_subdir,
        "--mask_ratio",   str(mask_ratio),
    ]
    print(f"\n>>> Running n_frames={n_frames} (all masked) ...")
    print(f"    Output: {out_subdir}")
    result = subprocess.run(cmd, check=True)
    return out_subdir


def collect_results(output_dir: str,
                    n_min: int, n_max: int,
                    exp1_ref: str | None) -> list:
    results = []

    # optional: include exp1 all-masked n=10 baseline
    if exp1_ref and os.path.isfile(exp1_ref):
        m = read_metrics(exp1_ref)
        m['n_frames'] = 10
        m['label'] = 'Exp1 ref\n(n=10, all masked)'
        results.append(m)
        print(f"  Loaded Exp1 reference (n=10): CD={m['chamfer_distance']:.6f}")

    for n in range(n_min, n_max + 1):
        p = os.path.join(output_dir, f"frames_{n:02d}", "metrics.txt")
        if not os.path.isfile(p):
            print(f"  [skip] missing {p}")
            continue
        m = read_metrics(p)
        m['n_frames'] = n
        m['label'] = str(n)
        results.append(m)

    results.sort(key=lambda x: x['n_frames'])
    return results


def plot(results: list, out_path: str):
    n_frames = [r['n_frames'] for r in results]
    cd       = [r['chamfer_distance'] for r in results]
    cd_pred  = [r.get('cd_pred_to_gt', None) for r in results]
    cd_gt    = [r.get('cd_gt_to_pred', None) for r in results]

    # split reference point (n=10) from exp2 points
    ref_idx  = [i for i, r in enumerate(results) if r['n_frames'] == 10]
    exp2_idx = [i for i, r in enumerate(results) if r['n_frames'] != 10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: symmetric CD ─────────────────────────────────────────
    ax = axes[0]
    exp2_x = [n_frames[i] for i in exp2_idx]
    exp2_y = [cd[i] for i in exp2_idx]
    ax.plot(exp2_x, exp2_y, marker='o', linewidth=2.2, markersize=8,
            color='#2196F3', markerfacecolor='white', markeredgewidth=2.5,
            label='Exp2 (all masked)')

    if ref_idx:
        rx, ry = n_frames[ref_idx[0]], cd[ref_idx[0]]
        ax.scatter([rx], [ry], s=100, color='#FF5722', zorder=5,
                   label=f'Exp1 ref  n=10  CD={ry:.4f}')
        ax.axhline(y=ry, color='#FF5722', linestyle='--', linewidth=1.2,
                   alpha=0.6)

    for x, y in zip(n_frames, cd):
        ax.annotate(f'{y:.4f}', (x, y),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8, color='#333')

    ax.set_xticks(n_frames)
    ax.set_xlabel('Total Number of Frames (all masked)', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Symmetric Chamfer Distance', fontsize=12, fontweight='bold')
    ax.set_ylim(min(cd) - 0.04, max(cd) + 0.09)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Right: directional CDs ─────────────────────────────────────
    ax2 = axes[1]
    if any(v is not None for v in cd_pred):
        ax2.plot(n_frames, cd_pred, marker='s', linewidth=2, markersize=7,
                 color='#E91E63', label='CD pred → GT')
        ax2.plot(n_frames, cd_gt, marker='^', linewidth=2, markersize=7,
                 color='#4CAF50', label='CD GT → pred')
        ax2.set_xticks(n_frames)
        ax2.set_xlabel('Total Number of Frames (all masked)', fontsize=12)
        ax2.set_ylabel('Chamfer Distance', fontsize=12)
        ax2.set_title('Directional Chamfer Distances', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle(
        'DUSt3R Reconstruction Quality vs. Number of Frames  (all masked)\n'
        'Teddybear · 101_11758_21048 · mask_ratio=0.25 · num_patches=3',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")

    # ── Print table ─────────────────────────────────────────────────
    print()
    print(f"{'n_frames':>10}  {'CD':>12}  {'CD pred→GT':>12}  {'CD GT→pred':>12}")
    print('-' * 52)
    for r in results:
        p = r.get('cd_pred_to_gt', float('nan'))
        g = r.get('cd_gt_to_pred', float('nan'))
        print(f"{r['n_frames']:>10}  {r['chamfer_distance']:>12.6f}"
              f"  {p:>12.6f}  {g:>12.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence_dir',  default=SEQ_DIR)
    parser.add_argument('--masked_dir',    default=MASKED_DIR)
    parser.add_argument('--output_dir',    default=OUTPUT_DIR)
    parser.add_argument('--n_frames_min',  type=int, default=11)
    parser.add_argument('--n_frames_max',  type=int, default=20)
    parser.add_argument('--mask_ratio',    type=float, default=MASK_RATIO)
    parser.add_argument('--exp1_ref',      default=EXP1_SWEEP,
                        help='Path to Exp1 all-masked metrics.txt for reference')
    parser.add_argument('--plot_only',     action='store_true',
                        help='Skip inference, just re-plot existing results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.plot_only:
        print("=" * 60)
        print("Experiment 2: All-masked frames sweep")
        print(f"  Sequence : {args.sequence_dir}")
        print(f"  Masked   : {args.masked_dir}")
        print(f"  n_frames : {args.n_frames_min} → {args.n_frames_max} (all masked)")
        print(f"  mask_ratio={args.mask_ratio}")
        print("=" * 60)

        for n in range(args.n_frames_min, args.n_frames_max + 1):
            run_one(n, args.sequence_dir, args.masked_dir,
                    args.output_dir, args.mask_ratio,
                    dust3r_dir="dust3r")

    results = collect_results(args.output_dir,
                              args.n_frames_min, args.n_frames_max,
                              args.exp1_ref)
    if not results:
        print("No results found. Run without --plot_only first.")
        return

    out_png = os.path.join(args.output_dir, 'chamfer_vs_n_frames.png')
    plot(results, out_png)


if __name__ == '__main__':
    main()
