"""
Plot Chamfer Distance vs. number of masked frames from the masking sweep.

Usage:
    python scripts/plot_sweep_results.py
    python scripts/plot_sweep_results.py --sweep_dir outputs/dust3r/sweep --out results.png
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_metrics(metrics_path: str) -> dict:
    """Parse a metrics.txt file and return a dict of key→float."""
    data = {}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                try:
                    data[key.strip()] = float(val.strip().replace(',', ''))
                except ValueError:
                    pass
    return data


def collect_results(sweep_dir: str) -> list:
    """Scan sweep_dir for masked_Xof10 subdirs and collect metrics."""
    results = []
    for entry in sorted(os.listdir(sweep_dir)):
        if not entry.startswith('masked_'):
            continue
        metrics_file = os.path.join(sweep_dir, entry, 'metrics.txt')
        if not os.path.isfile(metrics_file):
            print(f"  [skip] no metrics.txt in {entry}")
            continue
        m = read_metrics(metrics_file)
        try:
            n_masked = int(entry.split('_')[1].split('of')[0])
        except Exception:
            continue
        m['n_masked'] = n_masked
        results.append(m)
    results.sort(key=lambda x: x['n_masked'])
    return results


def plot(results: list, out_path: str, n_frames: int = 10):
    n_masked = [r['n_masked'] for r in results]
    cd       = [r['chamfer_distance'] for r in results]
    cd_pred  = [r.get('cd_pred_to_gt', None) for r in results]
    cd_gt    = [r.get('cd_gt_to_pred', None) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: symmetric CD ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(n_masked, cd, marker='o', linewidth=2.2, markersize=8,
            color='#2196F3', markerfacecolor='white', markeredgewidth=2.5,
            label='Chamfer Distance (symmetric)')
    ax.axhline(y=cd[0], color='gray', linestyle='--', linewidth=1.2,
               alpha=0.7, label=f'Baseline  CD={cd[0]:.4f}')

    for x, y in zip(n_masked, cd):
        ax.annotate(f'{y:.4f}', (x, y),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8, color='#333')

    pct_labels = [f'{i}\n({i * 100 // n_frames}%)' for i in n_masked]
    ax.set_xticks(n_masked)
    ax.set_xticklabels(pct_labels, fontsize=9)
    ax.set_xlabel('Number of Masked Frames  (% of total)', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Symmetric Chamfer Distance', fontsize=12, fontweight='bold')
    ax.set_ylim(min(cd) - 0.04, max(cd) + 0.09)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Right: directional CDs ─────────────────────────────────────
    ax2 = axes[1]
    if any(v is not None for v in cd_pred):
        ax2.plot(n_masked, cd_pred, marker='s', linewidth=2, markersize=7,
                 color='#E91E63', label='CD pred → GT')
        ax2.plot(n_masked, cd_gt, marker='^', linewidth=2, markersize=7,
                 color='#4CAF50', label='CD GT → pred')
        ax2.set_xticks(n_masked)
        ax2.set_xticklabels(pct_labels, fontsize=9)
        ax2.set_xlabel('Number of Masked Frames  (% of total)', fontsize=12)
        ax2.set_ylabel('Chamfer Distance', fontsize=12)
        ax2.set_title('Directional Chamfer Distances', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle(
        'DUSt3R Reconstruction Quality vs. Number of Masked Frames\n'
        'Teddybear · 101_11758_21048 · mask_ratio=0.25 · num_patches=3',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {out_path}")

    # ── Print results table ────────────────────────────────────────
    print()
    print(f"{'n_masked':>10}  {'CD':>12}  {'CD pred→GT':>12}  {'CD GT→pred':>12}")
    print('-' * 52)
    for r in results:
        p = r.get('cd_pred_to_gt', float('nan'))
        g = r.get('cd_gt_to_pred', float('nan'))
        print(f"{r['n_masked']:>10}  {r['chamfer_distance']:>12.6f}"
              f"  {p:>12.6f}  {g:>12.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', default='outputs/dust3r/sweep',
                        help='Directory containing masked_Xof10 subdirs')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: <sweep_dir>/chamfer_vs_masked.png)')
    parser.add_argument('--n_frames', type=int, default=10,
                        help='Total number of frames used per run')
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.sweep_dir, 'chamfer_vs_masked.png')

    results = collect_results(args.sweep_dir)
    if not results:
        print(f"No results found in {args.sweep_dir}")
        return

    print(f"Found {len(results)} sweep runs.")
    plot(results, out_path, n_frames=args.n_frames)


if __name__ == '__main__':
    main()
