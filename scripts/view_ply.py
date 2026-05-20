"""
Simple Gradio viewer for PLY point clouds.
Converts 1-3 PLY files to GLB and displays interactively.

Usage (single):
    python scripts/view_ply.py \
        --ply outputs/colmap/clean/teddybear_101_11758_21048/frames_070/predicted.ply \
        --port 7863

Usage (compare two):
    python scripts/view_ply.py \
        --ply  outputs/dust3r/grid_mask_exp/mask_10pct/bottle_34_1397_4376/frames_02/predicted.ply \
        --ply2 outputs/dust3r/grid_mask_exp/mask_10pct/bottle_34_1397_4376/frames_20/predicted.ply \
        --port 7863

Usage (compare three):
    python scripts/view_ply.py \
        --ply  outputs/a/predicted.ply \
        --ply2 outputs/b/predicted.ply \
        --ply3 outputs/c/predicted.ply \
        --port 7863
"""

import argparse
import os
import socket
import tempfile

import gradio
import numpy as np
import trimesh
from plyfile import PlyData


def ply_to_glb(ply_path: str, out_glb: str, max_pts: int = 200_000):
    """Load a PLY point cloud and save as GLB for the 3D viewer."""
    data = PlyData.read(ply_path)
    v = data["vertex"]
    xyz = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1).astype(np.float32)

    if "red" in v.data.dtype.names:
        rgb = np.stack([np.array(v["red"]),
                        np.array(v["green"]),
                        np.array(v["blue"])], axis=1).astype(np.uint8)
    else:
        z = xyz[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-8)
        cmap = trimesh.visual.color.interpolate(z_norm, color_map="viridis")
        rgb = (cmap[:, :3] * 255).astype(np.uint8)

    if len(xyz) > max_pts:
        idx = np.random.default_rng(42).choice(len(xyz), max_pts, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    pc = trimesh.PointCloud(vertices=xyz, colors=rgb)
    scene = trimesh.Scene([pc])
    scene.export(out_glb)
    print(f"Exported {len(xyz)} points → {out_glb}")
    return out_glb


def short_label(ply_path: str) -> str:
    parts = ply_path.rstrip("/").split("/")
    return " / ".join(parts[-3:]) if len(parts) >= 3 else ply_path


def pick_listen_port(preferred: int, span: int = 100) -> int:
    """Return `preferred` if free, else the next free port in [preferred, preferred+span)."""
    for p in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port in {preferred}–{preferred + span - 1}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply",  required=True, help="Path to first .ply file")
    parser.add_argument("--ply2", default=None,  help="Path to second .ply file (optional)")
    parser.add_argument("--ply3", default=None,  help="Path to third .ply file (optional)")
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()

    def make_glb(ply_path):
        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        tmp.close()
        ply_to_glb(ply_path, tmp.name)
        return tmp.name

    glb1 = make_glb(args.ply)
    label1 = short_label(args.ply)

    if args.ply2:
        glb2 = make_glb(args.ply2)
        label2 = short_label(args.ply2)
    if args.ply3:
        glb3 = make_glb(args.ply3)
        label3 = short_label(args.ply3)

    with gradio.Blocks(title="Point Cloud Viewer") as demo:
        if args.ply2 and args.ply3:
            gradio.HTML('<h2 style="text-align:center">DUSt3R Point Cloud Comparison (3 clouds)</h2>')
            with gradio.Row():
                gradio.Model3D(value=glb1, label=label1)
                gradio.Model3D(value=glb2, label=label2)
                gradio.Model3D(value=glb3, label=label3)
        elif args.ply2:
            gradio.HTML('<h2 style="text-align:center">DUSt3R Point Cloud Comparison (2 clouds)</h2>')
            with gradio.Row():
                gradio.Model3D(value=glb1, label=label1)
                gradio.Model3D(value=glb2, label=label2)
        else:
            gradio.HTML(f'<h2 style="text-align:center">Point Cloud — {label1}</h2>')
            gradio.Model3D(value=glb1, label="Rotate / zoom with mouse")

    port = pick_listen_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is busy; using {port} instead.")

    demo.launch(share=False, server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
