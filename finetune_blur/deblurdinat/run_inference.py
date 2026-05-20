"""
Inference: DeblurDiNAT + DUSt3R with Chamfer vs GT point cloud.

Finetuning in this repo uses DUSt3R ViT-L **224** (`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth`)
and `--resolution 224`; default `--image_size` here is **224** to match.

Usage (baseline DUSt3R, no deblurring):
  python finetune_blur/deblurdinat/run_inference.py \
    --sequence_dir /path/to/seq \
    --dust3r_dir ~/project_Hayk_Minasyan/dust3r \
    --dust3r_ckpt /path/to/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \
    --output_dir /path/to/output \
    --image_size 224 --n_frames 20 --device cuda

Usage (finetuned DeblurDiNAT + DUSt3R):
  ... --image_size 224 --pipeline deblur_finetuned \
    --finetuned_weights /path/to/checkpoint-best-val.pth \
    --deblurdinat_repo ~/project_Hayk_Minasyan/DeblurDiNAT
"""

import argparse
import os
import sys
import numpy as np
import torch
from scipy.spatial import KDTree


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
    return np.asarray(mesh.vertices if hasattr(mesh, "vertices") else mesh.points,
                      dtype=np.float32)


def compute_chamfer(pred, gt):
    tree_gt = KDTree(gt)
    tree_pred = KDTree(pred)
    d_p2g, _ = tree_gt.query(pred)
    d_g2p, _ = tree_pred.query(gt)
    cd_p2g = float(np.mean(d_p2g ** 2))
    cd_g2p = float(np.mean(d_g2p ** 2))
    return 0.5 * (cd_p2g + cd_g2p), cd_p2g, cd_g2p


def frame_list(images_dir, n_frames):
    imgs = sorted(
        os.path.join(images_dir, x) for x in os.listdir(images_dir)
        if x.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not imgs:
        raise RuntimeError(f"No images in {images_dir}")
    if n_frames >= len(imgs):
        return imgs
    idx = np.linspace(0, len(imgs) - 1, n_frames).round().astype(int)
    return [imgs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--dust3r_dir", required=True)
    ap.add_argument("--dust3r_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=224,
                    help="must match DUSt3R ckpt / finetuning (224 for ViT-L 224 linear)")
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--pipeline",
        choices=["dust3r", "deblur_pretrained", "deblur_finetuned"],
        default="dust3r",
        help="dust3r | DeblurDiNAT (GoPro init) | finetuned joint ckpt",
    )
    ap.add_argument("--is_finetuned", action="store_true",
                    help="deprecated: same as --pipeline deblur_finetuned")
    ap.add_argument("--finetuned_weights", default=None,
                    help="checkpoint-best-val.pth (for deblur_finetuned)")
    ap.add_argument("--deblurdinat_repo", default=None)
    ap.add_argument("--deblurdinat_pretrained_ckpt", default=None,
                    help="DeblurDiNATL.pth (GoPro); required for deblur_pretrained")
    ap.add_argument("--freeze", default="deblurdinat_only",
                    choices=["deblurdinat_only", "deblurdinat_and_decoder", "all"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.is_finetuned:
        args.pipeline = "deblur_finetuned"

    os.makedirs(args.output_dir, exist_ok=True)
    proj_dir = os.path.dirname(args.dust3r_dir.rstrip("/"))
    for p in [proj_dir, args.dust3r_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    device = torch.device(args.device)
    verbose = not args.quiet

    def _load_ckpt(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    if args.pipeline == "dust3r":
        from dust3r.model import load_model
        model = load_model(args.dust3r_ckpt, device="cpu", verbose=verbose).to(device).eval()
    else:
        from finetune_blur.deblurdinat.model import build_model
        dinit = args.deblurdinat_pretrained_ckpt if args.pipeline == "deblur_pretrained" else None
        model = build_model(
            dust3r_ckpt=args.dust3r_ckpt,
            deblurdinat_repo=args.deblurdinat_repo,
            deblurdinat_weights=dinit,
            device="cpu",
            freeze=args.freeze,
        )
        if args.pipeline == "deblur_finetuned":
            if not args.finetuned_weights:
                raise SystemExit("--finetuned_weights required for deblur_finetuned")
            ckpt = _load_ckpt(args.finetuned_weights)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

    images_dir = os.path.join(args.sequence_dir, "images")
    paths = frame_list(images_dir, args.n_frames)
    imgs = load_images(paths, size=args.image_size)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=verbose)

    mode = (GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2
            else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=300, schedule="linear", lr=0.01)

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
    print(f"Saved: {pred_ply} ({len(pred_pts):,} points)")

    gt_ply = os.path.join(args.sequence_dir, "pointcloud.ply")
    if os.path.isfile(gt_ply):
        gt_pts = load_ply_xyz(gt_ply)
        cd, cd_p2g, cd_g2p = compute_chamfer(pred_pts, gt_pts)
        metrics_path = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"pipeline: {args.pipeline}\n")
            f.write(f"n_pred_points: {len(pred_pts)}\n")
            f.write(f"n_gt_points: {len(gt_pts)}\n")
            f.write(f"chamfer_distance: {cd:.8f}\n")
            f.write(f"cd_pred_to_gt: {cd_p2g:.8f}\n")
            f.write(f"cd_gt_to_pred: {cd_g2p:.8f}\n")
        print(f"CD={cd:.6f}  (p2g={cd_p2g:.6f}, g2p={cd_g2p:.6f})")
    else:
        print(f"GT not found: {gt_ply}")


if __name__ == "__main__":
    main()
