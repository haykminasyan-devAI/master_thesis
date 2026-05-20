"""
Experiment 4: Same sweep as Exp3 (n_frames 21-40), but with mask_ratio=0.50
and num_patches=6.  Plots a line graph (n_frames vs CD) and overlays the
Exp3 curve (mask_ratio=0.25) for direct comparison.

Usage (after inference is done):
    python scripts/exp4.py
    python scripts/exp4.py --exp3_dir outputs/dust3r/exp3 \
                            --exp4_dir outputs/dust3r/exp4 \
                            --n_frames_min 21 --n_frames_max 40
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp3_dir',      default='outputs/dust3r/exp3')
    parser.add_argument('--exp4_dir',      default='outputs/dust3r/exp4')
    parser.add_argument('--n_frames_min',  type=int, default=2)
    parser.add_argument('--n_frames_max',  type=int, default=40)
    parser.add_argument('--out',           default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.exp4_dir, 'chamfer_vs_n_frames.png')
    os.makedirs(args.exp4_dir, exist_ok=True)

    exp3 = collect(args.exp3_dir, args.n_frames_min, args.n_frames_max)
    exp4 = collect(args.exp4_dir, args.n_frames_min, args.n_frames_max)

    if not exp3 and not exp4:
        print("No results found. Run inference first.")
        return

    # ── Print table ───────────────────────────────────────────────
    print(f"\n{'n_frames':>10}  {'Exp3 CD (25%)':>14}  {'Exp4 CD (50%)':>14}")
    print('-' * 44)
    all_n = sorted(set(r['n_frames'] for r in exp3 + exp4))
    exp3_map = {r['n_frames']: r['chamfer_distance'] for r in exp3}
    exp4_map = {r['n_frames']: r['chamfer_distance'] for r in exp4}
    for n in all_n:
        e3 = f"{exp3_map[n]:.6f}" if n in exp3_map else '   —   '
        e4 = f"{exp4_map[n]:.6f}" if n in exp4_map else '   —   '
        print(f"{n:>10}  {e3:>14}  {e4:>14}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: symmetric CD
    ax = axes[0]
    if exp3:
        ax.plot([r['n_frames'] for r in exp3],
                [r['chamfer_distance'] for r in exp3],
                marker='o', linewidth=2.2, markersize=7,
                color='#2196F3', markerfacecolor='white', markeredgewidth=2,
                label='Exp3: mask=25%, patches=3')
    if exp4:
        ax.plot([r['n_frames'] for r in exp4],
                [r['chamfer_distance'] for r in exp4],
                marker='^', linewidth=2.2, markersize=7,
                color='#E91E63', markerfacecolor='white', markeredgewidth=2,
                label='Exp4: mask=50%, patches=6')

    all_cd = [r['chamfer_distance'] for r in exp3 + exp4]
    ax.set_xlabel('Total Number of Frames (all masked)', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Symmetric Chamfer Distance', fontsize=12, fontweight='bold')
    ax.set_ylim(min(all_cd) - 0.05, max(all_cd) + 0.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_n)

    # Right: directional CDs for exp4
    ax2 = axes[1]
    if exp4 and all(r.get('cd_pred_to_gt') is not None for r in exp4):
        ax2.plot([r['n_frames'] for r in exp4],
                 [r.get('cd_pred_to_gt', float('nan')) for r in exp4],
                 marker='s', linewidth=2, markersize=7,
                 color='#E91E63', label='Exp4 CD pred → GT')
        ax2.plot([r['n_frames'] for r in exp4],
                 [r.get('cd_gt_to_pred', float('nan')) for r in exp4],
                 marker='^', linewidth=2, markersize=7,
                 color='#FF9800', label='Exp4 CD GT → pred')
        if exp3 and all(r.get('cd_pred_to_gt') is not None for r in exp3):
            ax2.plot([r['n_frames'] for r in exp3],
                     [r.get('cd_pred_to_gt', float('nan')) for r in exp3],
                     marker='s', linewidth=2, markersize=7, linestyle='--',
                     color='#2196F3', label='Exp3 CD pred → GT')
            ax2.plot([r['n_frames'] for r in exp3],
                     [r.get('cd_gt_to_pred', float('nan')) for r in exp3],
                     marker='^', linewidth=2, markersize=7, linestyle='--',
                     color='#4CAF50', label='Exp3 CD GT → pred')
        ax2.set_xlabel('Total Number of Frames (all masked)', fontsize=12)
        ax2.set_ylabel('Chamfer Distance', fontsize=12)
        ax2.set_title('Directional Chamfer Distances', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(all_n)
    else:
        axes[1].set_visible(False)

    fig.suptitle(
        'DUSt3R Reconstruction Quality vs. Number of Frames (all masked)\n'
        'Teddybear · 101_11758_21048 · mask=25% (Exp3) vs mask=50% (Exp4)',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")


if __name__ == '__main__':
    main()
