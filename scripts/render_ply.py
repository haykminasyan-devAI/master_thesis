"""
Render a PLY point cloud from multiple viewpoints and save as PNG.
No display or CDN needed — output opens directly in the IDE.

Usage:
    python scripts/render_ply.py --ply outputs/dust3r/sweep/masked_3of10/predicted.ply
    python scripts/render_ply.py --ply_a <pred.ply> --ply_b <gt.ply> --out out.png
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


def load_ply(path: str):
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header.append(line)
            if line == 'end_header':
                break
        n_verts, is_binary = 0, False
        for line in header:
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            if 'binary' in line:
                is_binary = True

        if is_binary:
            dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                           ('r','u1'),('g','u1'),('b','u1')])
            data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
            xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
            rgb = np.stack([data['r'], data['g'], data['b']], axis=1).astype(np.float32) / 255.
        else:
            rows = [list(map(float, f.readline().split())) for _ in range(n_verts)]
            arr  = np.array(rows, dtype=np.float32)
            xyz  = arr[:, :3]
            rgb  = arr[:, 3:6] / 255. if arr.shape[1] >= 6 else \
                   np.full((n_verts, 3), 0.6, dtype=np.float32)
    return xyz, rgb


def subsample(xyz, rgb, max_pts=80_000):
    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        return xyz[idx], rgb[idx]
    return xyz, rgb


def render_views(xyz, rgb, title, axes_list, views):
    """Render point cloud from multiple viewpoints into given axes."""
    # centre the cloud
    centre = xyz.mean(axis=0)
    xyz = xyz - centre
    scale = np.percentile(np.linalg.norm(xyz, axis=1), 95)
    xyz = xyz / (scale + 1e-8)

    for ax, (elev, azim) in zip(axes_list, views):
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=rgb, s=0.3, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.set_axis_off()
        ax.set_title(f'{title}\nelev={elev} azim={azim}', fontsize=8, pad=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply',   default=None, help='Single PLY to render')
    parser.add_argument('--ply_a', default=None, help='First PLY  (e.g. predicted)')
    parser.add_argument('--ply_b', default=None, help='Second PLY (e.g. GT)')
    parser.add_argument('--label_a', default='Predicted')
    parser.add_argument('--label_b', default='Ground Truth')
    parser.add_argument('--out',   default=None, help='Output PNG path')
    parser.add_argument('--max_pts', type=int, default=80_000)
    args = parser.parse_args()

    VIEWS = [(30, 0), (30, 90), (30, 180), (30, 270), (80, 45), (-30, 135)]

    if args.ply:
        # single PLY — 2 rows: top row=predicted, bottom row=GT side
        xyz, rgb = subsample(*load_ply(args.ply), args.max_pts)
        label    = os.path.basename(os.path.dirname(args.ply))
        out_path = args.out or args.ply.replace('.ply', '_views.png')

        n = len(VIEWS)
        fig = plt.figure(figsize=(4 * n, 4))
        axes = [fig.add_subplot(1, n, i+1, projection='3d') for i in range(n)]
        render_views(xyz, rgb, label, axes, VIEWS)
        fig.suptitle(label, fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {out_path}")

    else:
        # two PLYs side by side
        assert args.ply_a and args.ply_b, "Provide --ply or both --ply_a and --ply_b"
        xyz_a, rgb_a = subsample(*load_ply(args.ply_a), args.max_pts)
        xyz_b, rgb_b = subsample(*load_ply(args.ply_b), args.max_pts)

        out_dir  = os.path.dirname(args.ply_a)
        out_path = args.out or os.path.join(out_dir, 'comparison_views.png')

        n = len(VIEWS)
        fig = plt.figure(figsize=(4 * n, 8))
        axes_a = [fig.add_subplot(2, n, i+1,   projection='3d') for i in range(n)]
        axes_b = [fig.add_subplot(2, n, i+1+n, projection='3d') for i in range(n)]

        render_views(xyz_a, rgb_a, args.label_a, axes_a, VIEWS)
        render_views(xyz_b, rgb_b, args.label_b, axes_b, VIEWS)

        fig.suptitle(f'{args.label_a}  vs  {args.label_b}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
