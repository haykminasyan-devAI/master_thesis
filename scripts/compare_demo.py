"""
Side-by-side Gradio demo comparing two DUSt3R reconstructions + GT.

Default: compares sweep masked_3of10 vs masked_4of10 with GT.

Usage:
    python scripts/compare_demo.py
    python scripts/compare_demo.py --ply_a outputs/dust3r/sweep/masked_3of10/predicted.ply \
                                   --ply_b outputs/dust3r/sweep/masked_4of10/predicted.ply \
                                   --port 7870
"""

import os
import argparse
import numpy as np

# ── defaults ──────────────────────────────────────────────────────
PLY_A     = "outputs/dust3r/sweep/masked_3of10/predicted.ply"
PLY_B     = "outputs/dust3r/sweep/masked_4of10/predicted.ply"
GT_PLY    = "data/co3d/teddybear/101_11758_21048/pointcloud.ply"
LABEL_A   = "n_masked=3  (3 masked, 7 clean)"
LABEL_B   = "n_masked=4  (4 masked, 6 clean)"
LABEL_GT  = "Ground Truth"
OUT_DIR   = "outputs/dust3r/compare_demo"
PORT      = 7870
MAX_PTS   = 500_000


def load_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLY file and return (xyz, rgb) arrays."""
    import struct

    with open(path, 'rb') as f:
        # parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # find vertex count and format
        n_verts = 0
        is_binary = False
        props = []
        for line in header_lines:
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            if 'binary' in line:
                is_binary = True
            if line.startswith('property'):
                props.append(line.split()[-1])

        if is_binary:
            # binary little-endian: x,y,z (float32) + r,g,b (uint8)
            dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                           ('r','u1'),('g','u1'),('b','u1')])
            data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
            xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
            rgb = np.stack([data['r'], data['g'], data['b']], axis=1)
        else:
            # ascii fallback
            rows = []
            for _ in range(n_verts):
                rows.append(list(map(float, f.readline().split())))
            arr = np.array(rows, dtype=np.float32)
            xyz = arr[:, :3]
            rgb = arr[:, 3:6].astype(np.uint8) if arr.shape[1] >= 6 else \
                  np.full((n_verts, 3), 180, dtype=np.uint8)

    return xyz, rgb


def ply_to_glb(ply_path: str, out_path: str, max_pts: int = MAX_PTS) -> str:
    import trimesh
    xyz, rgb = load_ply(ply_path)
    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    alpha = np.full((len(xyz), 1), 255, dtype=np.uint8)
    rgba  = np.concatenate([rgb.astype(np.uint8), alpha], axis=1)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=xyz, colors=rgba))
    scene.export(out_path)
    print(f"  Exported {len(xyz):,} pts → {out_path}")
    return out_path


def load_gt_glb(gt_ply: str, out_path: str, max_pts: int = MAX_PTS) -> str:
    import trimesh
    try:
        pc = trimesh.load(gt_ply)
        xyz = np.array(pc.vertices)
        if hasattr(pc, 'colors') and pc.colors is not None and len(pc.colors):
            rgb = np.array(pc.colors)[:, :3].astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), [100, 180, 255], dtype=np.uint8)
    except Exception:
        xyz, rgb = load_ply(gt_ply)

    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    alpha = np.full((len(xyz), 1), 255, dtype=np.uint8)
    rgba  = np.concatenate([rgb.astype(np.uint8), alpha], axis=1)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=xyz, colors=rgba))
    scene.export(out_path)
    print(f"  GT exported {len(xyz):,} pts → {out_path}")
    return out_path


def launch_demo(glb_a: str, glb_b: str, glb_gt: str,
                label_a: str, label_b: str, label_gt: str, port: int):
    import gradio

    title = "DUSt3R Reconstruction Comparison"

    with gradio.Blocks(title=title) as demo:
        gradio.HTML(f'<h2 style="text-align:center">{title}</h2>')
        gradio.HTML(
            '<p style="text-align:center;color:#555">'
            'Teddybear · 101_11758_21048 · 10 frames · mask_ratio=0.25 · num_patches=3'
            '</p>'
        )
        with gradio.Row():
            with gradio.Column():
                gradio.HTML(f'<h3 style="text-align:center;color:#E53935">{label_a}</h3>')
                gradio.Model3D(value=glb_a, label=label_a)
            with gradio.Column():
                gradio.HTML(f'<h3 style="text-align:center;color:#F57C00">{label_b}</h3>')
                gradio.Model3D(value=glb_b, label=label_b)
            with gradio.Column():
                gradio.HTML(f'<h3 style="text-align:center;color:#1976D2">{label_gt}</h3>')
                gradio.Model3D(value=glb_gt, label=label_gt)

    print(f"\nDemo running at: http://localhost:{port}")
    print(f"SSH tunnel:  ssh -L {port}:localhost:{port} <cluster>")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, block=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_a',    default=PLY_A)
    parser.add_argument('--ply_b',    default=PLY_B)
    parser.add_argument('--gt_ply',   default=GT_PLY)
    parser.add_argument('--label_a',  default=LABEL_A)
    parser.add_argument('--label_b',  default=LABEL_B)
    parser.add_argument('--label_gt', default=LABEL_GT)
    parser.add_argument('--out_dir',  default=OUT_DIR)
    parser.add_argument('--port',     type=int, default=PORT)
    parser.add_argument('--max_pts',  type=int, default=MAX_PTS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Converting PLY → GLB ...")
    glb_a  = ply_to_glb(args.ply_a,  os.path.join(args.out_dir, "pred_a.glb"), args.max_pts)
    glb_b  = ply_to_glb(args.ply_b,  os.path.join(args.out_dir, "pred_b.glb"), args.max_pts)
    glb_gt = load_gt_glb(args.gt_ply, os.path.join(args.out_dir, "gt.glb"),    args.max_pts)

    print("\nLaunching Gradio demo ...")
    launch_demo(glb_a, glb_b, glb_gt,
                args.label_a, args.label_b, args.label_gt,
                args.port)


if __name__ == '__main__':
    main()
