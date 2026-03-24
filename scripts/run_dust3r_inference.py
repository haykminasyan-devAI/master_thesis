"""
Run DUSt3R inference on a CO3D sequence and compare against GT point cloud.
Supports mixing clean and masked frames to study the effect of occlusion.

Usage (clean baseline):
    python scripts/run_dust3r_inference.py \
        --sequence_dir data/co3d/teddybear/101_11758_21048 \
        --dust3r_dir   dust3r \
        --n_frames     10 \
        --output_dir   outputs/dust3r/teddybear_clean

Usage (masking experiment — 5 of 10 frames masked at 25%):
    python scripts/run_dust3r_inference.py \
        --sequence_dir data/co3d/teddybear/101_11758_21048 \
        --dust3r_dir   dust3r \
        --n_frames     10 \
        --n_masked     5 \
        --masked_dir   outputs/masked_frames/teddybear/101_11758_21048/mask_25pct \
        --mask_ratio   0.25 \
        --output_dir   outputs/dust3r/teddybear_masked5_25pct
"""

import os
import sys
import argparse
import numpy as np
import torch

# ── add dust3r to Python path ──────────────────────────────────────────────────
def setup_dust3r_path(dust3r_dir: str):
    dust3r_dir = os.path.abspath(dust3r_dir)
    sys.path.insert(0, dust3r_dir)
    croco_dir = os.path.join(dust3r_dir, "croco")
    sys.path.insert(0, croco_dir)
    sys.path.insert(0, os.path.join(croco_dir, "models"))

# ── frame selection ────────────────────────────────────────────────────────────
def select_frames(images_dir: str, n_frames: int) -> list[str]:
    """Return n_frames evenly-spaced frame filenames (just basenames, sorted)."""
    all_frames = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])
    if len(all_frames) <= n_frames:
        return all_frames
    indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
    return [all_frames[i] for i in indices]


def build_frame_list(images_dir: str, n_frames: int,
                     n_masked: int = 0, masked_dir: str = None) -> list[str]:
    """
    Build a mixed list of clean and masked frame paths.

    The same n_frames evenly-spaced frames are always selected.
    The first n_masked of them are loaded from masked_dir,
    the rest from images_dir (original clean frames).

    Args:
        images_dir : original images folder
        n_frames   : total frames to use (default 10)
        n_masked   : how many of those frames to replace with masked versions
        masked_dir : folder containing the masked frames (same filenames)
    """
    frame_names = select_frames(images_dir, n_frames)

    if n_masked == 0 or masked_dir is None:
        return [os.path.join(images_dir, f) for f in frame_names]

    assert 0 <= n_masked <= len(frame_names), \
        f"n_masked ({n_masked}) must be between 0 and n_frames ({len(frame_names)})"

    paths = []
    for i, fname in enumerate(frame_names):
        if i < n_masked:
            p = os.path.join(masked_dir, fname)
            assert os.path.isfile(p), f"Masked frame not found: {p}"
            paths.append(p)
        else:
            paths.append(os.path.join(images_dir, fname))
    return paths

# ── All metrics ────────────────────────────────────────────────────────────────
def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute all evaluation metrics between predicted and GT point clouds.

    Metrics:
      - Chamfer Distance (symmetric, pred→GT, GT→pred)
      - Hausdorff Distance (symmetric = max of the two directed distances)
      - F1 Score @ 1% of GT bounding-box diagonal
      - PSNR d1 (point-to-point, peak = GT bbox diagonal)

    pred, gt: (N,3) and (M,3) float32 numpy arrays
    """
    from scipy.spatial import KDTree

    pred = pred.astype(np.float32)
    gt   = gt.astype(np.float32)

    tree_gt   = KDTree(gt)
    tree_pred = KDTree(pred)

    dist_p2g, _ = tree_gt.query(pred)    # pred → GT  (L2 distances)
    dist_g2p, _ = tree_pred.query(gt)    # GT → pred  (L2 distances)

    # ── Chamfer Distance ──────────────────────────────────────────
    cd_pred_to_gt = float(np.mean(dist_p2g ** 2))
    cd_gt_to_pred = float(np.mean(dist_g2p ** 2))
    cd = (cd_pred_to_gt + cd_gt_to_pred) / 2.0

    # ── Hausdorff Distance ────────────────────────────────────────
    hausdorff_p2g = float(np.max(dist_p2g))
    hausdorff_g2p = float(np.max(dist_g2p))
    hausdorff     = max(hausdorff_p2g, hausdorff_g2p)

    # ── GT bounding-box diagonal (used as scale for F1 and PSNR) ─
    bbox_min = gt.min(axis=0)
    bbox_max = gt.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    # ── F1 Score @ 1% bbox diagonal ──────────────────────────────
    threshold = 0.01 * bbox_diag
    precision = float(np.mean(dist_p2g <= threshold))   # frac pred hits
    recall    = float(np.mean(dist_g2p <= threshold))   # frac GT hits
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # ── PSNR d1 (point-to-point) ──────────────────────────────────
    # MSE = mean of squared nearest-neighbour distances (both directions)
    mse_d1 = (np.mean(dist_p2g ** 2) + np.mean(dist_g2p ** 2)) / 2.0
    peak   = bbox_diag
    if mse_d1 > 0:
        psnr_d1 = 10.0 * np.log10(peak ** 2 / mse_d1)
    else:
        psnr_d1 = float('inf')

    return {
        # Chamfer
        "chamfer_distance": cd,
        "cd_pred_to_gt":    cd_pred_to_gt,
        "cd_gt_to_pred":    cd_gt_to_pred,
        # Hausdorff
        "hausdorff":        hausdorff,
        "hausdorff_p2g":    hausdorff_p2g,
        "hausdorff_g2p":    hausdorff_g2p,
        # F1
        "f1":               f1,
        "precision":        precision,
        "recall":           recall,
        "f1_threshold":     threshold,
        # PSNR d1
        "psnr_d1":          float(psnr_d1),
        "bbox_diag":        bbox_diag,
        # Point counts
        "n_pred_points":    len(pred),
        "n_gt_points":      len(gt),
    }


# keep old name as alias for backward compatibility
def chamfer_distance(pred, gt):
    return compute_all_metrics(pred, gt)

# ── save point cloud as binary .ply ───────────────────────────────────────────
def save_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
    assert points.shape[1] == 3
    colors = (colors * 255).clip(0, 255).astype(np.uint8) if colors.max() <= 1.0 else colors.astype(np.uint8)
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    # pack xyz (float32) + rgb (uint8) into a structured array and write at once
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["r"], arr["g"], arr["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())

# ── load GT point cloud from .ply ──────────────────────────────────────────────
def load_ply(path: str) -> np.ndarray:
    from plyfile import PlyData
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return pts

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DUSt3R inference + Chamfer Distance vs CO3D GT")
    parser.add_argument("--sequence_dir", required=True,
                        help="Path to a CO3D sequence folder (must contain images/ and pointcloud.ply)")
    parser.add_argument("--dust3r_dir",   required=True, help="Path to the cloned DUSt3R repo")
    parser.add_argument("--output_dir",   default="outputs/dust3r", help="Where to save results")
    parser.add_argument("--device",       default="cuda", help="cuda or cpu")
    # ── frame parameters ──────────────────────────────────────────────────────
    parser.add_argument("--n_frames",  type=int,   default=10,
                        help="Total number of frames to use (default: 10)")
    parser.add_argument("--n_masked",  type=int,   default=0,
                        help="How many of the n_frames to replace with masked versions (default: 0)")
    parser.add_argument("--masked_dir", type=str,  default=None,
                        help="Folder with masked frames (same filenames as images/). "
                             "Required when --n_masked > 0")
    parser.add_argument("--mask_ratio", type=float, default=0.25,
                        help="Mask ratio used when generating masked frames (for logging, default: 0.25)")
    # ── model parameters ──────────────────────────────────────────────────────
    parser.add_argument("--min_conf_thr", type=float, default=3.0,
                        help="Confidence threshold for point filtering (default: 3.0)")
    args = parser.parse_args()

    if args.n_masked > 0 and args.masked_dir is None:
        parser.error("--masked_dir is required when --n_masked > 0")

    setup_dust3r_path(args.dust3r_dir)

    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load model ─────────────────────────────────────────────────────────────
    print("Loading DUSt3R model ...")
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    model = model.to(args.device).eval()
    print("Model loaded.")

    # ── select frames ──────────────────────────────────────────────────────────
    images_dir  = os.path.join(args.sequence_dir, "images")
    frame_paths = build_frame_list(images_dir, args.n_frames,
                                   args.n_masked, args.masked_dir)

    print(f"\nUsing {len(frame_paths)} frames  "
          f"(masked: {args.n_masked}, clean: {len(frame_paths)-args.n_masked})")
    for i, p in enumerate(frame_paths):
        tag = "[MASKED]" if i < args.n_masked else "[clean] "
        print(f"  {tag} {os.path.basename(p)}")

    # ── run DUSt3R inference ───────────────────────────────────────────────────
    print("\nRunning DUSt3R inference ...")
    imgs = load_images(frame_paths, size=512)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=1)

    # ── global alignment ───────────────────────────────────────────────────────
    print("\nRunning global alignment ...")
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=args.device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=300, schedule="linear", lr=0.01)

    # ── extract point cloud ────────────────────────────────────────────────────
    pts3d = to_numpy(scene.get_pts3d())
    imgs_rgb = scene.imgs
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr)))
    masks = to_numpy(scene.get_masks())

    pred_pts = np.concatenate([p[m] for p, m in zip(pts3d, masks)], axis=0)
    pred_colors = np.concatenate([p[m] for p, m in zip(imgs_rgb, masks)], axis=0)

    # Filter out NaN/Inf values
    valid = np.isfinite(pred_pts).all(axis=1)
    pred_pts = pred_pts[valid]
    pred_colors = pred_colors[valid]

    print(f"\nPredicted point cloud: {len(pred_pts):,} points")

    if len(pred_pts) == 0:
        print("WARNING: predicted point cloud is empty after filtering. Skipping.")
        return

    # ── save predicted .ply ────────────────────────────────────────────────────
    pred_ply_path = os.path.join(args.output_dir, "predicted.ply")
    save_ply(pred_pts, pred_colors, pred_ply_path)
    print(f"Saved predicted point cloud to: {pred_ply_path}")

    # ── load GT point cloud ────────────────────────────────────────────────────
    gt_ply_path = os.path.join(args.sequence_dir, "pointcloud.ply")
    if not os.path.isfile(gt_ply_path):
        print(f"\nWARNING: GT point cloud not found at {gt_ply_path}. Skipping evaluation.")
        return

    gt_pts = load_ply(gt_ply_path)
    print(f"GT point cloud: {len(gt_pts):,} points")

    # ── compute metrics ────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics = compute_all_metrics(pred_pts, gt_pts)

    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)
    print(f"  Chamfer Distance (CD):       {metrics['chamfer_distance']:.6f}")
    print(f"  CD pred→GT:                  {metrics['cd_pred_to_gt']:.6f}")
    print(f"  CD GT→pred:                  {metrics['cd_gt_to_pred']:.6f}")
    print(f"  Hausdorff Distance:          {metrics['hausdorff']:.6f}")
    print(f"  Hausdorff pred→GT:           {metrics['hausdorff_p2g']:.6f}")
    print(f"  Hausdorff GT→pred:           {metrics['hausdorff_g2p']:.6f}")
    print(f"  F1 Score (thr={metrics['f1_threshold']:.4f}):    {metrics['f1']:.6f}")
    print(f"  Precision:                   {metrics['precision']:.6f}")
    print(f"  Recall:                      {metrics['recall']:.6f}")
    print(f"  PSNR d1:                     {metrics['psnr_d1']:.4f} dB")
    print(f"  BBox diagonal (GT):          {metrics['bbox_diag']:.6f}")
    print(f"  Predicted points:            {metrics['n_pred_points']:,}")
    print(f"  GT points:                   {metrics['n_gt_points']:,}")
    print("="*55)

    # ── save metrics ───────────────────────────────────────────────────────────
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"sequence:          {args.sequence_dir}\n")
        f.write(f"n_frames:          {args.n_frames}\n")
        f.write(f"n_masked:          {args.n_masked}\n")
        f.write(f"n_clean:           {args.n_frames - args.n_masked}\n")
        f.write(f"mask_ratio:        {args.mask_ratio}\n")
        f.write(f"n_pred_points:     {metrics['n_pred_points']}\n")
        f.write(f"n_gt_points:       {metrics['n_gt_points']}\n")
        f.write(f"chamfer_distance:  {metrics['chamfer_distance']:.8f}\n")
        f.write(f"cd_pred_to_gt:     {metrics['cd_pred_to_gt']:.8f}\n")
        f.write(f"cd_gt_to_pred:     {metrics['cd_gt_to_pred']:.8f}\n")
        f.write(f"hausdorff:         {metrics['hausdorff']:.8f}\n")
        f.write(f"hausdorff_p2g:     {metrics['hausdorff_p2g']:.8f}\n")
        f.write(f"hausdorff_g2p:     {metrics['hausdorff_g2p']:.8f}\n")
        f.write(f"f1:                {metrics['f1']:.8f}\n")
        f.write(f"precision:         {metrics['precision']:.8f}\n")
        f.write(f"recall:            {metrics['recall']:.8f}\n")
        f.write(f"f1_threshold:      {metrics['f1_threshold']:.8f}\n")
        f.write(f"psnr_d1:           {metrics['psnr_d1']:.6f}\n")
        f.write(f"bbox_diag:         {metrics['bbox_diag']:.8f}\n")
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
