"""
Automated DUSt3R demo with GT point cloud side by side.

Automatically selects frames from a CO3D sequence, runs DUSt3R inference,
and launches a Gradio viewer with two separate 3D windows:
  left  → DUSt3R predicted scene
  right → Ground Truth point cloud

Usage (on a GPU node):
    cd /home/asds/project_Hayk_Minasyan
    conda run -n co3d_env python scripts/auto_demo.py \
        --sequence_dir data/co3d/teddybear/101_11758_21048 \
        --n_frames 10 \
        --port 7860

Then on your LOCAL machine run:
    ssh -L 7860:<compute-node>:7860 asds@<login-node-ip>

Open:  http://localhost:7860
"""

import os
import sys
import argparse
import numpy as np
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUST3R    = os.path.join(ROOT, "dust3r")
CROCO     = os.path.join(DUST3R, "croco")
sys.path.insert(0, DUST3R)
sys.path.insert(0, CROCO)
sys.path.insert(0, os.path.join(CROCO, "models"))


def select_frames(images_dir: str, n: int) -> list:
    frames = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if len(frames) <= n:
        return frames
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def build_frame_list(images_dir: str, n_frames: int,
                     n_masked: int = 0, masked_dir: str = None) -> list:
    """Select n_frames evenly spaced; first n_masked come from masked_dir."""
    all_names = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    indices = np.linspace(0, len(all_names) - 1, n_frames, dtype=int)
    frame_names = [all_names[i] for i in indices]

    if n_masked == 0 or masked_dir is None:
        return [os.path.join(images_dir, f) for f in frame_names]

    paths = []
    for i, fname in enumerate(frame_names):
        if i < n_masked:
            p = os.path.join(masked_dir, fname)
            assert os.path.isfile(p), f"Masked frame not found: {p}"
            paths.append(p)
            print(f"  [MASKED] {fname}")
        else:
            paths.append(os.path.join(images_dir, fname))
            print(f"  [clean]  {fname}")
    return paths


def load_gt_cloud(pointcloud_ply: str):
    """Return (N,3) xyz and (N,3) rgb arrays from a .ply file."""
    from plyfile import PlyData
    plydata = PlyData.read(pointcloud_ply)
    v = plydata["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    try:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    except Exception:
        # fallback: solid sky-blue colour
        rgb = np.tile(np.array([80, 160, 255], dtype=np.uint8), (len(xyz), 1))
    return xyz, rgb


def build_gt_glb(outdir: str, gt_xyz: np.ndarray, gt_rgb: np.ndarray,
                 max_gt_pts: int = 500_000) -> str:
    """Export GT point cloud as a standalone GLB file."""
    import trimesh

    if len(gt_xyz) > max_gt_pts:
        idx = np.random.choice(len(gt_xyz), max_gt_pts, replace=False)
        gt_xyz, gt_rgb = gt_xyz[idx], gt_rgb[idx]

    alpha = np.full((len(gt_xyz), 1), 255, dtype=np.uint8)
    gt_rgba = np.concatenate([gt_rgb.astype(np.uint8), alpha], axis=1)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=gt_xyz, colors=gt_rgba))

    gt_path = os.path.join(outdir, "gt.glb")
    scene.export(gt_path)
    print(f"GT scene saved to: {gt_path}")
    return gt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir",  required=True,
                        help="Path to a CO3D sequence (contains images/ and pointcloud.ply)")
    parser.add_argument("--n_frames",      type=int, default=10)
    parser.add_argument("--port",          type=int, default=7860)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--model_name",    default="DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    parser.add_argument("--image_size",    type=int, default=512, choices=[512, 224])
    parser.add_argument("--niter",         type=int, default=300)
    parser.add_argument("--schedule",      default="cosine")
    parser.add_argument("--min_conf_thr",  type=float, default=3.0)
    parser.add_argument("--max_gt_pts",    type=int, default=500_000,
                        help="Max GT points to embed in the GLB (default 500k)")
    parser.add_argument("--serve_only",    action="store_true",
                        help="Skip inference, just serve existing GLB files")
    parser.add_argument("--n_masked",      type=int, default=0,
                        help="Number of frames to load from masked_dir (first N frames)")
    parser.add_argument("--masked_dir",    default=None,
                        help="Directory with pre-masked frames")
    args = parser.parse_args()

    os.chdir(ROOT)

    # ── validate paths ────────────────────────────────────────────────────────
    images_dir = os.path.join(args.sequence_dir, "images")
    gt_ply     = os.path.join(args.sequence_dir, "pointcloud.ply")
    assert os.path.isdir(images_dir), f"images/ not found in {args.sequence_dir}"
    assert os.path.isfile(gt_ply),    f"pointcloud.ply not found in {args.sequence_dir}"

    # ── select frames ─────────────────────────────────────────────────────────
    print(f"\nSelected {args.n_frames} frames  (masked: {args.n_masked}, clean: {args.n_frames - args.n_masked}):")
    frames = build_frame_list(images_dir, args.n_frames,
                              n_masked=args.n_masked,
                              masked_dir=args.masked_dir)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading DUSt3R model ({args.model_name}) on {args.device} ...")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.demo  import get_reconstructed_scene, set_print_with_timestamp
    set_print_with_timestamp()
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/" + args.model_name).to(args.device)
    model.eval()
    print("Model loaded.\n")

    # ── output directory ──────────────────────────────────────────────────────
    outdir = os.path.join(ROOT, "outputs", "demo",
                          os.path.basename(args.sequence_dir))
    os.makedirs(outdir, exist_ok=True)

    pred_glb     = os.path.join(outdir, "scene.glb")
    gt_glb_path  = os.path.join(outdir, "gt.glb")
    gallery_imgs = []
    scene        = None

    if args.serve_only:
        assert os.path.isfile(pred_glb),    f"scene.glb not found in {outdir} — run without --serve_only first"
        assert os.path.isfile(gt_glb_path), f"gt.glb not found in {outdir} — run without --serve_only first"
        print(f"Serve-only mode: loading existing GLB files from {outdir}")
        gt_glb = gt_glb_path
    else:
        print("Running DUSt3R inference + global alignment ...")
        scene, pred_glb, gallery_imgs = get_reconstructed_scene(
        outdir, model, args.device,
        silent=False,
        image_size=args.image_size,
        filelist=frames,
        schedule=args.schedule,
        niter=args.niter,
        min_conf_thr=args.min_conf_thr,
        as_pointcloud=True,      # point cloud mode, not mesh
        mask_sky=False,
        clean_depth=True,
        transparent_cams=False,
        cam_size=0.05,
        scenegraph_type="complete",
        winsize=1,
        refid=0,
    )
        print(f"\nPredicted scene: {pred_glb}")

        # build GT GLB
        print(f"Loading GT point cloud: {gt_ply}")
        gt_xyz, gt_rgb = load_gt_cloud(gt_ply)
        print(f"  {len(gt_xyz):,} GT points")
        gt_glb = build_gt_glb(outdir, gt_xyz, gt_rgb, max_gt_pts=args.max_gt_pts)

    # ── launch Gradio viewer ──────────────────────────────────────────────────
    import gradio
    import functools
    from dust3r.demo import get_3D_model_from_scene

    category   = os.path.basename(os.path.dirname(args.sequence_dir))
    seq_name   = os.path.basename(args.sequence_dir)
    page_title = f"DUSt3R + GT — {category} · {seq_name} · {args.n_frames} frames"

    with gradio.Blocks(title=page_title) as demo_app:
        gradio.HTML(f'<h2 style="text-align:center">{page_title}</h2>')

        scene_state = gradio.State(scene)

        with gradio.Row():
            with gradio.Column():
                gradio.HTML('<h3 style="text-align:center">DUSt3R Predicted</h3>')
                outmodel_pred = gradio.Model3D(value=pred_glb, label="Predicted")

            with gradio.Column():
                gradio.HTML('<h3 style="text-align:center">Ground Truth</h3>')
                outmodel_gt = gradio.Model3D(value=gt_glb, label="Ground Truth")

        with gradio.Row():
            with gradio.Column():
                gradio.HTML("<h4>Adjust prediction</h4>")
                min_conf_thr_sl    = gradio.Slider(label="min_conf_thr", value=args.min_conf_thr,
                                                   minimum=1.0, maximum=20, step=0.1)
                cam_size_sl        = gradio.Slider(label="cam_size", value=0.05,
                                                   minimum=0.001, maximum=0.1, step=0.001)
                as_pointcloud_cb   = gradio.Checkbox(value=True,  label="As pointcloud")
                mask_sky_cb        = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth_cb     = gradio.Checkbox(value=True,  label="Clean-up depthmaps")
                transparent_cams_cb= gradio.Checkbox(value=False, label="Transparent cameras")
                regen_btn          = gradio.Button("Re-export prediction")

        gradio.HTML("<h3>Per-frame: RGB · Depth · Confidence</h3>")
        gradio.Gallery(value=gallery_imgs, label="frames", columns=3, height="auto")

        def regen(scene_s, conf, pc, sky, depth, tcam, csize):
            new_pred = get_3D_model_from_scene(
                outdir, False, scene_s, conf, pc, sky, depth, tcam, csize)
            return new_pred if new_pred else pred_glb

        regen_btn.click(
            fn=regen,
            inputs=[scene_state, min_conf_thr_sl, as_pointcloud_cb,
                    mask_sky_cb, clean_depth_cb, transparent_cams_cb, cam_size_sl],
            outputs=outmodel_pred,
        )

    print("\n" + "="*60)
    print(f"  Viewer ready!")
    print(f"")
    print(f"  On your LOCAL machine run:")
    print(f"    ssh -L {args.port}:$(hostname):{args.port} asds@<login-ip>")
    print(f"")
    print(f"  Then open:  http://localhost:{args.port}")
    print("="*60 + "\n")

    demo_app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
