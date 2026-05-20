"""
Visualize COLMAP sparse point cloud vs GT point cloud for multiple categories.

Creates a grid of subplots: rows = categories, cols = selected n_frames values.
Each cell shows the predicted point cloud (blue) overlaid on GT point cloud (gray).

Usage:
    python scripts/expCOLMAP/visualize_colmap.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from plyfile import PlyData

_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SCRIPTS_DIR)

# ── config ─────────────────────────────────────────────────────────────────────
RESULTS_ROOT = "outputs/colmap/masked_25pct"
CO3D_BASE    = "data/co3d"

SEQUENCES = {
    "teddybear": "101_11758_21048",
    "bottle":    "34_1397_4376",
    "hydrant":   "106_12648_23157",
    "cup":       "12_100_593",
    "toybus":    "111_13154_25988",
    "toytrain":  "104_12352_22039",
}

SHOW_N_FRAMES = [50, 100, 150]   # columns in the grid
MAX_POINTS    = 8_000            # subsample for speed


# ── helpers ────────────────────────────────────────────────────────────────────

def load_ply_xyz(path: str, max_pts: int = MAX_POINTS) -> np.ndarray | None:
    """Load XYZ from a binary or ASCII PLY file; subsample if needed."""
    if not os.path.isfile(path):
        return None
    try:
        data = PlyData.read(path)
        v = data["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
        return pts
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}")
        return None


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


def scatter3d(ax, pts: np.ndarray, color, alpha=0.25, s=0.5, label=None):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=color, s=s, alpha=alpha, label=label, rasterized=True)


def set_equal_axes(ax, pts_list):
    """Set equal axis limits across all point sets."""
    all_pts = np.concatenate([p for p in pts_list if p is not None and len(p) > 0])
    mn, mx = all_pts.min(0), all_pts.max(0)
    center = (mn + mx) / 2
    half   = (mx - mn).max() / 2 * 1.1
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    categories = list(SEQUENCES.keys())
    n_cols = len(SHOW_N_FRAMES)
    n_rows = len(categories)

    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
    fig.patch.set_facecolor("#1C1C1C")

    for row_idx, cat in enumerate(categories):
        seq_id  = SEQUENCES[cat]
        gt_path = os.path.join(CO3D_BASE, cat, seq_id, "pointcloud.ply")
        gt_pts  = load_ply_xyz(gt_path, max_pts=6000)

        for col_idx, n in enumerate(SHOW_N_FRAMES):
            ax = fig.add_subplot(n_rows, n_cols,
                                 row_idx * n_cols + col_idx + 1,
                                 projection="3d")
            ax.set_facecolor("#1C1C1C")

            # load predicted point cloud
            pred_path = os.path.join(RESULTS_ROOT, f"{cat}_{seq_id}",
                                     f"frames_{n:02d}", "predicted.ply")
            pred_pts  = load_ply_xyz(pred_path)

            metrics_path = os.path.join(RESULTS_ROOT, f"{cat}_{seq_id}",
                                        f"frames_{n:02d}", "metrics.txt")
            cd = read_chamfer(metrics_path)

            # plot
            if gt_pts is not None:
                scatter3d(ax, gt_pts, color="#888888", alpha=0.15, s=0.4, label="GT")
            if pred_pts is not None:
                scatter3d(ax, pred_pts, color="#4FC3F7", alpha=0.6, s=1.2, label="COLMAP")
                if gt_pts is not None:
                    set_equal_axes(ax, [gt_pts, pred_pts])
            else:
                ax.text2D(0.5, 0.5, "no reconstruction",
                          ha="center", va="center",
                          transform=ax.transAxes,
                          color="white", fontsize=9)

            # title
            cd_str = f"CD={cd:.2f}" if cd is not None else "CD=N/A"
            ax.set_title(f"{cat}  n={n}\n{cd_str}",
                         color="white", fontsize=9, pad=4)

            # clean up axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("#333333")
            ax.yaxis.pane.set_edgecolor("#333333")
            ax.zaxis.pane.set_edgecolor("#333333")
            ax.view_init(elev=20, azim=45)

        print(f"  [{row_idx+1}/{n_rows}] {cat} done")

    # shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=8, label="GT point cloud", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4FC3F7",
               markersize=8, label="COLMAP prediction", linestyle="None"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=11, framealpha=0.3, edgecolor="#555",
               labelcolor="white", facecolor="#1C1C1C",
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("COLMAP Sparse Reconstruction vs GT  |  mask_ratio=25%\n"
                 "Gray = GT, Blue = COLMAP  |  Chamfer Distance shown per cell",
                 color="white", fontsize=13, y=1.01)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(RESULTS_ROOT, "colmap_pointcloud_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
