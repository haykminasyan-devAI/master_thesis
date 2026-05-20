"""
Side-by-side video: input frames (left) + rotating 3D point cloud (right).

Usage:
    python scripts/ply_to_video.py \
        --ply        outputs/dust3r/grid_mask_exp/mask_10pct/teddybear_101_11758_21048/frames_20/predicted.ply \
        --frames_dir outputs/demo/teddybear_n20_mask10pct \
        --output     outputs/demo/teddybear_n20_mask10pct.mp4 \
        --title      "Teddybear — 20 frames, 10% mask"
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from plyfile import PlyData
from PIL import Image


def load_ply(path, max_pts=80_000):
    data = PlyData.read(path)
    v = data["vertex"]
    xyz = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1).astype(np.float32)
    if "red" in v.data.dtype.names:
        rgb = np.stack([np.array(v["red"]),
                        np.array(v["green"]),
                        np.array(v["blue"])], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.ones((len(xyz), 3)) * 0.6
    if len(xyz) > max_pts:
        idx = np.random.default_rng(42).choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def load_frames(frames_dir):
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([os.path.join(frames_dir, f)
                    for f in os.listdir(frames_dir)
                    if f.lower().endswith(exts)])
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply",        required=True)
    parser.add_argument("--frames_dir", required=True, help="Folder with input images")
    parser.add_argument("--output",     required=True, help="Output .mp4 path")
    parser.add_argument("--title",      default="Point Cloud")
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--seconds",    type=int,   default=15)
    parser.add_argument("--init_elev",  type=float, default=20,  help="Starting elevation angle (default 20)")
    parser.add_argument("--init_azim",  type=float, default=0,   help="Starting azimuth angle (default 0)")
    args = parser.parse_args()

    total = args.fps * args.seconds

    print(f"Loading PLY: {args.ply} ...")
    xyz, rgb = load_ply(args.ply)
    center = np.median(xyz, axis=0)
    std = np.std(xyz, axis=0)
    mask = np.all(np.abs(xyz - center) < 3 * std, axis=1)
    xyz, rgb = xyz[mask], rgb[mask]
    print(f"  {len(xyz):,} points after outlier removal")

    print(f"Loading frames from: {args.frames_dir}")
    frames = load_frames(args.frames_dir)
    print(f"  {len(frames)} images loaded")

    # ── figure: left=image, right=3D ─────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=0.90, bottom=0, wspace=0.02)

    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.set_axis_off()

    ax3d = fig.add_subplot(1, 2, 2, projection="3d", facecolor="white")
    ax3d.set_axis_off()

    ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                 c=rgb, s=0.4, linewidths=0, depthshade=True)

    for setter, data in zip([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim],
                             [xyz[:, 0], xyz[:, 1], xyz[:, 2]]):
        lo, hi = data.min(), data.max()
        pad = (hi - lo) * 0.05
        setter(lo - pad, hi + pad)

    fig.suptitle(args.title, fontsize=14, fontweight="bold", color="black", y=0.97)
    ax_img.set_title("Input frames", fontsize=11, color="#333333")
    ax3d.set_title("DUSt3R 3D reconstruction", fontsize=11, color="#333333")

    # initial image
    im_handle = ax_img.imshow(frames[0])

    # elevation path over the video (relative to init_elev)
    e0 = args.init_elev
    def get_elev(t):
        if t < 0.25:
            return e0 + 30 * (t / 0.25)
        elif t < 0.50:
            return (e0 + 30) - 30 * ((t - 0.25) / 0.25)
        elif t < 0.75:
            return e0 - 30 * ((t - 0.50) / 0.25)
        else:
            return (e0 - 30) + 30 * ((t - 0.75) / 0.25)

    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    print(f"Rendering {total} frames ({args.seconds}s) → {args.output}")

    with writer.saving(fig, args.output, dpi=120):
        for i in range(total):
            t = i / total
            # cycle through input frames
            fi = int(t * len(frames)) % len(frames)
            im_handle.set_data(frames[fi])
            # rotate point cloud
            ax3d.view_init(elev=get_elev(t), azim=args.init_azim + 360 * t * 2)
            writer.grab_frame()
            if i % args.fps == 0:
                print(f"  {i // args.fps}s / {args.seconds}s")

    plt.close()
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
