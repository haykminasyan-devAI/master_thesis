"""
DUSt3R inference for the angle range experiment.
Limits the frame pool to the first X% of the sequence (simulating limited viewpoint coverage).
Only computes Chamfer Distance.

Usage:
    python outputs/dust3r/angle_range_exp/inference.py \
        --sequence_dir /mnt/weka/hminasyan/data/co3d/teddybear/101_11758_21048 \
        --dust3r_dir   dust3r \
        --n_frames     5 \
        --frame_pool_pct 25.0 \
        --output_dir   /mnt/weka/hminasyan/outputs/dust3r/angle_range_exp/range_0_90deg/teddybear_101_11758_21048/frames_05
"""

import os
import sys
import argparse
import numpy as np
import torch


def setup_dust3r_path(dust3r_dir: str):
    dust3r_dir = os.path.abspath(dust3r_dir)
    sys.path.insert(0, dust3r_dir)
    croco_dir = os.path.join(dust3r_dir, "croco")
    sys.path.insert(0, croco_dir)
    sys.path.insert(0, os.path.join(croco_dir, "models"))


def select_frames(images_dir: str, n_frames: int, frame_pool_pct: float) -> list:
    """Return n_frames evenly-spaced frames from the first frame_pool_pct% of the sequence."""
    all_frames = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])
    pool_end = max(2, int(len(all_frames) * frame_pool_pct / 100.0))
    pool = all_frames[:pool_end]
    if len(pool) <= n_frames:
        return pool
    indices = np.linspace(0, len(pool) - 1, n_frames, dtype=int)
    return [pool[i] for i in indices]


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> dict:
    from scipy.spatial import KDTree
    pred = pred.astype(np.float32)
    gt   = gt.astype(np.float32)
    dist_p2g, _ = KDTree(gt).query(pred)
    dist_g2p, _ = KDTree(pred).query(gt)
    cd = (float(np.mean(dist_p2g ** 2)) + float(np.mean(dist_g2p ** 2))) / 2.0
    return {
        "chamfer_distance": cd,
        "cd_pred_to_gt":    float(np.mean(dist_p2g ** 2)),
        "cd_gt_to_pred":    float(np.mean(dist_g2p ** 2)),
        "n_pred_points":    len(pred),
        "n_gt_points":      len(gt),
    }


def load_ply(path: str) -> np.ndarray:
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def save_ply(points: np.ndarray, colors: np.ndarray, path: str):
    colors = (colors * 255).clip(0, 255).astype(np.uint8) if colors.max() <= 1.0 else colors.astype(np.uint8)
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = points[:,0], points[:,1], points[:,2]
    arr["r"], arr["g"], arr["b"] = colors[:,0], colors[:,1], colors[:,2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence_dir",   required=True)
    parser.add_argument("--dust3r_dir",     required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--n_frames",       type=int,   required=True)
    parser.add_argument("--frame_pool_pct", type=float, required=True,
                        help="Use first X%% of frames (16.67=0-60deg, 25=0-90deg, 50=0-180deg)")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--min_conf_thr",   type=float, default=3.0)
    args = parser.parse_args()

    setup_dust3r_path(args.dust3r_dir)

    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading DUSt3R model ...")
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    model = model.to(args.device).eval()

    images_dir = os.path.join(args.sequence_dir, "images")
    selected   = select_frames(images_dir, args.n_frames, args.frame_pool_pct)
    frame_paths = [os.path.join(images_dir, f) for f in selected]

    print(f"Using {len(frame_paths)} frames from first {args.frame_pool_pct:.1f}% of sequence:")
    for p in frame_paths:
        print(f"  {os.path.basename(p)}")

    print("\nRunning DUSt3R inference ...")
    imgs   = load_images(frame_paths, size=512)
    pairs  = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=1)

    print("Running global alignment ...")
    mode  = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=args.device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=300, schedule="linear", lr=0.01)

    pts3d    = to_numpy(scene.get_pts3d())
    imgs_rgb = scene.imgs
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr)))
    masks    = to_numpy(scene.get_masks())

    pred_pts    = np.concatenate([p[m] for p, m in zip(pts3d, masks)], axis=0)
    pred_colors = np.concatenate([p[m] for p, m in zip(imgs_rgb, masks)], axis=0)
    valid       = np.isfinite(pred_pts).all(axis=1)
    pred_pts, pred_colors = pred_pts[valid], pred_colors[valid]

    print(f"Predicted point cloud: {len(pred_pts):,} points")
    if len(pred_pts) == 0:
        print("WARNING: empty point cloud. Skipping.")
        return

    save_ply(pred_pts, pred_colors, os.path.join(args.output_dir, "predicted.ply"))

    gt_ply_path = os.path.join(args.sequence_dir, "pointcloud.ply")
    if not os.path.isfile(gt_ply_path):
        print(f"WARNING: GT not found at {gt_ply_path}. Skipping evaluation.")
        return

    gt_pts = load_ply(gt_ply_path)
    print(f"GT point cloud: {len(gt_pts):,} points")

    print("Computing Chamfer Distance ...")
    m = chamfer_distance(pred_pts, gt_pts)

    print(f"\n  chamfer_distance: {m['chamfer_distance']:.6f}")
    print(f"  cd_pred_to_gt:    {m['cd_pred_to_gt']:.6f}")
    print(f"  cd_gt_to_pred:    {m['cd_gt_to_pred']:.6f}")

    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"sequence:          {args.sequence_dir}\n")
        f.write(f"n_frames:          {args.n_frames}\n")
        f.write(f"frame_pool_pct:    {args.frame_pool_pct}\n")
        f.write(f"n_pred_points:     {m['n_pred_points']}\n")
        f.write(f"n_gt_points:       {m['n_gt_points']}\n")
        f.write(f"chamfer_distance:  {m['chamfer_distance']:.8f}\n")
        f.write(f"cd_pred_to_gt:     {m['cd_pred_to_gt']:.8f}\n")
        f.write(f"cd_gt_to_pred:     {m['cd_gt_to_pred']:.8f}\n")
    print(f"Saved → {args.output_dir}/metrics.txt")


if __name__ == "__main__":
    main()
