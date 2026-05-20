"""
DUSt3R / IFAN+DUSt3R inference on synthetic defocused CO3D frames; Chamfer vs GT point cloud.

Pipelines:
  dust3r          — vanilla DUSt3R on defocused RGB
  ifan_pretrained — IFAN (pretrained ckpt) + frozen DUSt3R
  ifan_finetuned  — IFAN+DUSt3R joint finetuned checkpoint (IFAN weights updated)
"""
import argparse
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from scipy.spatial import KDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
for p in [os.path.join(PROJECT_DIR, "dust3r"), PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _disk_kernel(radius: int) -> np.ndarray:
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    m = (xx * xx + yy * yy) <= (r * r)
    k = m.astype(np.float32)
    return k / max(k.sum(), 1.0)


def apply_defocus_rgb(rgb: np.ndarray, radius: int) -> np.ndarray:
    k = _disk_kernel(radius)
    return cv2.filter2D(rgb, -1, k, borderType=cv2.BORDER_REFLECT)


def save_ply(points, colors, out_path):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def load_ply_xyz(path):
    import trimesh

    mesh = trimesh.load(path, process=False)
    return np.asarray(
        mesh.vertices if hasattr(mesh, "vertices") else mesh.points, dtype=np.float32
    )


def compute_chamfer(pred, gt):
    tree_gt = KDTree(gt)
    tree_pred = KDTree(pred)
    d_p2g, _ = tree_gt.query(pred)
    d_g2p, _ = tree_pred.query(gt)
    cd_p2g = float(np.mean(d_p2g**2))
    cd_g2p = float(np.mean(d_g2p**2))
    return 0.5 * (cd_p2g + cd_g2p), cd_p2g, cd_g2p


def frame_list(images_dir, n_frames):
    imgs = sorted(
        os.path.join(images_dir, x)
        for x in os.listdir(images_dir)
        if x.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not imgs:
        raise RuntimeError(f"No images in {images_dir}")
    if n_frames >= len(imgs):
        return imgs
    idx = np.linspace(0, len(imgs) - 1, n_frames).round().astype(int)
    return [imgs[i] for i in idx]


def _resolve_co3d_sequence_paths(co3d_root: str, category: str, seq_id: str, co3d_raw_root: str = None):
    """
    Processed CO3D often only has pointcloud.ply (+ metadata) under co3d_processed_*;
    RGB frames stay in the raw `data/co3d/.../images` tree.
    """
    from pathlib import Path

    root = Path(co3d_root)
    seq_proc = root / category / seq_id
    img_proc = seq_proc / "images"
    gt_proc = seq_proc / "pointcloud.ply"

    candidates_img = []
    if img_proc.is_dir():
        candidates_img.append(str(img_proc))

    raw = Path(co3d_raw_root) if co3d_raw_root else root.parent / "co3d"
    img_raw = raw / category / seq_id / "images"
    if img_raw.is_dir():
        candidates_img.append(str(img_raw))

    if not candidates_img:
        raise FileNotFoundError(
            f"No images/ found under {img_proc} or {img_raw}. "
            f"Set --co3d_raw_root to your raw CO3D root (e.g. .../data/co3d)."
        )
    src_img_dir = candidates_img[0]

    gt_ply = None
    if gt_proc.is_file():
        gt_ply = str(gt_proc)
    else:
        gt_raw = raw / category / seq_id / "pointcloud.ply"
        if gt_raw.is_file():
            gt_ply = str(gt_raw)
    if not gt_ply:
        raise FileNotFoundError(
            f"No pointcloud.ply under {seq_proc} or {raw / category / seq_id}"
        )

    return src_img_dir, gt_ply


def prepare_defocus_sequence(
    co3d_root: str,
    category: str,
    seq_id: str,
    defocus_radius: int,
    n_frames: int,
    co3d_raw_root: str = None,
):
    """Return path to temp dir with images/ (defocused) and pointcloud.ply copy."""
    src_img_dir, gt_ply = _resolve_co3d_sequence_paths(
        co3d_root, category, seq_id, co3d_raw_root=co3d_raw_root
    )

    tmp = tempfile.mkdtemp(prefix=f"defocus_eval_{category}_{seq_id}_")
    img_out = os.path.join(tmp, "images")
    os.makedirs(img_out, exist_ok=True)

    paths = frame_list(src_img_dir, n_frames)
    for i, p in enumerate(paths):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_d = apply_defocus_rgb(rgb, defocus_radius)
        out_p = os.path.join(img_out, f"frame_{i:06d}.jpg")
        cv2.imwrite(out_p, cv2.cvtColor(rgb_d, cv2.COLOR_RGB2BGR))

    shutil.copy2(gt_ply, os.path.join(tmp, "pointcloud.ply"))
    return tmp


def _load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d_root", required=True)
    ap.add_argument(
        "--co3d_raw_root",
        default=None,
        help="Raw CO3D root with images/ (default: <parent of co3d_root>/co3d)",
    )
    ap.add_argument("--category", required=True)
    ap.add_argument("--seq_id", required=True)
    ap.add_argument("--defocus_radius", type=int, default=6)
    ap.add_argument("--dust3r_dir", required=True)
    ap.add_argument("--dust3r_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--pipeline",
        choices=["dust3r", "ifan_pretrained", "ifan_finetuned"],
        default="dust3r",
    )
    ap.add_argument("--ifan_repo", default=None)
    ap.add_argument("--ifan_ckpt", default=None)
    ap.add_argument("--finetuned_weights", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    proj_dir = os.path.dirname(args.dust3r_dir.rstrip("/"))
    for p in [proj_dir, args.dust3r_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    device = torch.device(args.device)
    verbose = not args.quiet

    seq_dir = prepare_defocus_sequence(
        args.co3d_root,
        args.category,
        args.seq_id,
        args.defocus_radius,
        args.n_frames,
        co3d_raw_root=args.co3d_raw_root,
    )
    try:
        if args.pipeline == "dust3r":
            from dust3r.model import load_model

            model = load_model(args.dust3r_ckpt, device="cpu", verbose=verbose).to(device).eval()
        else:
            if not args.ifan_repo or not args.ifan_ckpt:
                raise SystemExit("--ifan_repo and --ifan_ckpt required for IFAN pipelines")
            from finetune_defocus.model_ifan_dust3r import build_model

            model = build_model(
                dust3r_ckpt=args.dust3r_ckpt,
                ifan_repo=args.ifan_repo,
                ifan_ckpt=args.ifan_ckpt,
                device="cpu",
                freeze="ifan_only",
            )
            if args.pipeline == "ifan_finetuned":
                if not args.finetuned_weights:
                    raise SystemExit("--finetuned_weights required for ifan_finetuned")
                ckpt = _load_ckpt(args.finetuned_weights)
                state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                model.load_state_dict(state, strict=False)
            model = model.to(device).eval()

        images_dir = os.path.join(seq_dir, "images")
        paths = sorted(
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        imgs = load_images(paths, size=args.image_size)
        pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, verbose=verbose)

        mode = (
            GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2
            else GlobalAlignerMode.PairViewer
        )
        scene = global_aligner(output, device=device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            scene.compute_global_alignment(
                init="mst", niter=300, schedule="linear", lr=0.01
            )

        pts3d = to_numpy(scene.get_pts3d())
        masks = to_numpy(scene.get_masks())
        imgs_rgb = scene.imgs

        all_pts, all_cols = [], []
        for i in range(len(pts3d)):
            m = masks[i]
            p = pts3d[i][m]
            im = imgs_rgb[i]
            if hasattr(im, "permute"):
                im = im.permute(1, 2, 0).detach().cpu().numpy()
            elif im.ndim == 3 and im.shape[0] in (1, 3) and im.shape[-1] not in (1, 3):
                im = np.transpose(im, (1, 2, 0))
            c = (im[m] * 255.0).clip(0, 255).astype(np.uint8)
            all_pts.append(p)
            all_cols.append(c)

        pred_pts = np.concatenate(all_pts, axis=0)
        pred_cols = np.concatenate(all_cols, axis=0)
        pred_ply = os.path.join(args.output_dir, "pred_pointcloud.ply")
        save_ply(pred_pts, pred_cols, pred_ply)

        gt_ply = os.path.join(seq_dir, "pointcloud.ply")
        gt_pts = load_ply_xyz(gt_ply)
        cd, cd_p2g, cd_g2p = compute_chamfer(pred_pts, gt_pts)
        metrics_path = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"pipeline: {args.pipeline}\n")
            f.write(f"category: {args.category}\n")
            f.write(f"seq_id: {args.seq_id}\n")
            f.write(f"defocus_radius: {args.defocus_radius}\n")
            f.write(f"chamfer_distance: {cd:.8f}\n")
            f.write(f"cd_pred_to_gt: {cd_p2g:.8f}\n")
            f.write(f"cd_gt_to_pred: {cd_g2p:.8f}\n")
        print(f"CD={cd:.8f} category={args.category} seq={args.seq_id} pipeline={args.pipeline}")
    finally:
        shutil.rmtree(seq_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
