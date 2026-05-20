"""
Run COLMAP SfM on n evenly-spaced masked CO3D frames and compare
the sparse point cloud against the CO3D GT using Chamfer distance.

The masked frames are already pre-generated in masked_dir.
We select n evenly-spaced frames from that directory, write an
image-list file, run COLMAP (feature extraction → matching → mapper),
align the sparse point cloud to the GT frame via Umeyama, then
compute Chamfer distance.

Usage:
    python scripts/expCOLMAP/run_colmap_inference.py \\
        --masked_dir   outputs/masked_frames/teddybear/101_11758_21048/mask_25pct \\
        --sequence_dir data/co3d/teddybear/101_11758_21048 \\
        --output_dir   outputs/colmap/masked_25pct/teddybear_101_11758_21048/frames_20 \\
        --n_frames     20 \\
        --gpu_index    0
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

# ── re-use helpers from the top-level run_colmap.py ───────────────────────────
_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SCRIPTS_DIR)
from run_colmap import (
    read_co3d_annotations,
    ndc_to_pixel_intrinsics,
    co3d_camera_center,
    read_colmap_points3d_bin,
    read_colmap_points3d_txt,
    read_colmap_images_bin,
    read_colmap_images_txt,
    umeyama,
    apply_similarity,
    run_cmd,
)
from run_dust3r_inference import compute_all_metrics, load_ply, save_ply


# ── frame selection ────────────────────────────────────────────────────────────

def select_frames(images_dir: str, n_frames: int) -> list:
    """Return n_frames evenly-spaced filenames from images_dir (sorted order)."""
    all_frames = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(all_frames) <= n_frames:
        return all_frames
    indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
    return [all_frames[i] for i in indices]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--masked_dir",   required=True,
                        help="Directory containing pre-generated masked frames")
    parser.add_argument("--sequence_dir", required=True,
                        help="CO3D sequence directory (contains pointcloud.ply)")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to write COLMAP workspace and results")
    parser.add_argument("--n_frames",     type=int, default=20,
                        help="Number of evenly-spaced frames to feed into COLMAP")
    parser.add_argument("--gpu_index",    type=int, default=0,
                        help="GPU index for SIFT (default: 0)")
    parser.add_argument("--skip_sfm",     action="store_true",
                        help="Skip SfM if already done (just re-run metrics)")
    args = parser.parse_args()

    metrics_txt = os.path.join(args.output_dir, "metrics.txt")
    if os.path.isfile(metrics_txt):
        print(f"[skip] already done: {metrics_txt}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    db_path    = os.path.join(args.output_dir, "colmap.db")
    sparse_dir = os.path.join(args.output_dir, "sparse")
    raw_ply    = os.path.join(args.output_dir, "predicted_raw.ply")
    aligned_ply= os.path.join(args.output_dir, "predicted.ply")
    images_abs = os.path.abspath(args.masked_dir)

    # ── CO3D camera intrinsics (shared across the whole sequence) ─────────────
    seq_name = os.path.basename(args.sequence_dir)
    cat_dir  = os.path.dirname(args.sequence_dir)
    ann_file = os.path.join(cat_dir, "frame_annotations.jgz")
    annotations = read_co3d_annotations(ann_file, seq_name)
    ann0 = annotations[0]
    H, W = ann0["image"]["size"]
    fx, fy, cx, cy = ndc_to_pixel_intrinsics(
        ann0["viewpoint"]["focal_length"],
        ann0["viewpoint"]["principal_point"],
        (H, W),
    )
    print(f"[n={args.n_frames}] camera {W}×{H}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # ── select n evenly-spaced frames ─────────────────────────────────────────
    selected = select_frames(images_abs, args.n_frames)
    actual_n = len(selected)
    print(f"[n={args.n_frames}] using {actual_n} frames from {images_abs}")

    if actual_n < 3:
        print("WARNING: fewer than 3 frames — skipping.")
        with open(metrics_txt, "w") as f:
            f.write(f"n_frames:  {args.n_frames}\nstatus:    skipped_too_few_frames\n")
        return

    # write image list for COLMAP (just filenames, relative to image_path)
    img_list_path = os.path.join(args.output_dir, "image_list.txt")
    with open(img_list_path, "w") as f:
        f.write("\n".join(selected) + "\n")

    # ── GT camera centers (used for Umeyama alignment later) ──────────────────
    co3d_centers = {}
    for ann in annotations:
        fname = os.path.basename(ann["image"]["path"])
        co3d_centers[fname] = co3d_camera_center(
            ann["viewpoint"]["R"], ann["viewpoint"]["T"]
        )

    # ── COLMAP SfM ────────────────────────────────────────────────────────────
    if not args.skip_sfm:

        # Step 1: Feature extraction (GPU SIFT, fixed PINHOLE intrinsics)
        run_cmd([
            "colmap", "feature_extractor",
            "--database_path",                   db_path,
            "--image_path",                      images_abs,
            "--image_list_path",                 img_list_path,
            "--ImageReader.single_camera",       "1",
            "--ImageReader.camera_model",        "PINHOLE",
            "--ImageReader.camera_params",       f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f}",
            "--FeatureExtraction.use_gpu",       "1",
            "--FeatureExtraction.gpu_index",     str(args.gpu_index),
            "--SiftExtraction.max_num_features", "8192",
        ], f"Step 1 – Feature extraction  (n={args.n_frames})")

        # Step 2: Sequential feature matching (GPU, loop detection disabled to avoid SIGSEGV)
        run_cmd([
            "colmap", "sequential_matcher",
            "--database_path",              db_path,
            "--FeatureMatching.use_gpu",    "1",
            "--FeatureMatching.gpu_index",  str(args.gpu_index),
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.loop_detection", "0",
        ], f"Step 2 – Sequential matching  (n={args.n_frames})")

        # Step 3: Sparse mapper
        os.makedirs(sparse_dir, exist_ok=True)
        try:
            run_cmd([
                "colmap", "mapper",
                "--database_path",                   db_path,
                "--image_path",                      images_abs,
                "--output_path",                     sparse_dir,
                "--Mapper.num_threads",              "4",
                "--Mapper.init_min_num_inliers",     "15",
                "--Mapper.abs_pose_min_num_inliers", "15",
                "--Mapper.ba_global_frames_ratio",   "1.1",
                "--Mapper.ba_global_points_ratio",   "1.1",
            ], f"Step 3 – Sparse mapper  (n={args.n_frames})")
        except subprocess.CalledProcessError as e:
            print(f"COLMAP mapper failed (n={args.n_frames}): {e}")
            with open(metrics_txt, "w") as f:
                f.write(f"n_frames:  {args.n_frames}\nstatus:    mapper_failed\n")
            return

        # pick the largest sub-model (by image count)
        subdirs = sorted(
            d for d in os.listdir(sparse_dir)
            if os.path.isdir(os.path.join(sparse_dir, d))
        )
        if not subdirs:
            print(f"No sub-model produced (n={args.n_frames}) — too few matches.")
            with open(metrics_txt, "w") as f:
                f.write(f"n_frames:  {args.n_frames}\nstatus:    no_model\n")
            return
        sparse_0 = os.path.join(sparse_dir, subdirs[0])

        # Step 4: Convert binary model to TXT
        run_cmd([
            "colmap", "model_converter",
            "--input_path",  sparse_0,
            "--output_path", sparse_0,
            "--output_type", "TXT",
        ], f"Step 4 – Model → TXT  (n={args.n_frames})")

    else:
        sparse_0 = os.path.join(sparse_dir, "0")
        print(f"Skipping SfM (--skip_sfm); using {sparse_0}")

    # ── read COLMAP 3D points ─────────────────────────────────────────────────
    p3d_txt = os.path.join(sparse_0, "points3D.txt")
    p3d_bin = os.path.join(sparse_0, "points3D.bin")
    if os.path.isfile(p3d_txt):
        pred_pts_raw, pred_colors = read_colmap_points3d_txt(p3d_txt)
    elif os.path.isfile(p3d_bin):
        pred_pts_raw, pred_colors = read_colmap_points3d_bin(p3d_bin)
    else:
        print(f"No points3D file in {sparse_0}")
        with open(metrics_txt, "w") as f:
            f.write(f"n_frames:  {args.n_frames}\nstatus:    no_points3D\n")
        return

    if len(pred_pts_raw) == 0:
        print(f"Empty point cloud (n={args.n_frames})")
        with open(metrics_txt, "w") as f:
            f.write(f"n_frames:  {args.n_frames}\nstatus:    empty_pointcloud\n")
        return

    print(f"COLMAP sparse: {len(pred_pts_raw):,} 3D points")
    save_ply(pred_pts_raw, pred_colors, raw_ply)

    # ── read COLMAP estimated camera centers ──────────────────────────────────
    img_txt = os.path.join(sparse_0, "images.txt")
    img_bin = os.path.join(sparse_0, "images.bin")
    if os.path.isfile(img_txt):
        colmap_centers = read_colmap_images_txt(img_txt)
    elif os.path.isfile(img_bin):
        colmap_centers = read_colmap_images_bin(img_bin)
    else:
        print(f"No images file in {sparse_0}")
        with open(metrics_txt, "w") as f:
            f.write(f"n_frames:  {args.n_frames}\nstatus:    no_images_file\n")
        return

    # ── Umeyama: align COLMAP frame → GT (CO3D) frame ─────────────────────────
    src_pts, dst_pts = [], []
    for fname, center in colmap_centers.items():
        if fname in co3d_centers:
            src_pts.append(center)
            dst_pts.append(co3d_centers[fname])
    matched = len(src_pts)
    print(f"Camera correspondences: {matched}/{actual_n} registered")

    if matched < 3:
        print(f"WARNING: only {matched} cameras registered — saving raw (unaligned) PLY.")
        shutil.copy(raw_ply, aligned_ply)
        pred_pts = pred_pts_raw
    else:
        scale, R_align, t_align = umeyama(np.array(src_pts), np.array(dst_pts))
        print(f"  Umeyama: scale={scale:.4f}")
        pred_pts = apply_similarity(pred_pts_raw, scale, R_align, t_align)
        save_ply(pred_pts, pred_colors, aligned_ply)
        print(f"Saved aligned PLY → {aligned_ply}")

    # ── GT point cloud + metrics ──────────────────────────────────────────────
    gt_ply = os.path.join(args.sequence_dir, "pointcloud.ply")
    gt_pts = load_ply(gt_ply)
    print(f"GT: {len(gt_pts):,} points")

    metrics = compute_all_metrics(pred_pts, gt_pts)

    print(f"\n{'='*60}")
    print(f"  COLMAP  n_frames={args.n_frames}  →  "
          f"Chamfer={metrics['chamfer_distance']:.6f}")
    print(f"{'='*60}\n")

    with open(metrics_txt, "w") as f:
        f.write(f"method:             COLMAP\n")
        f.write(f"n_frames:           {args.n_frames}\n")
        f.write(f"actual_frames_used: {actual_n}\n")
        f.write(f"cameras_registered: {matched}\n")
        f.write(f"status:             ok\n")
        f.write(f"n_pred_points:      {metrics['n_pred_points']}\n")
        f.write(f"n_gt_points:        {metrics['n_gt_points']}\n")
        f.write(f"chamfer_distance:   {metrics['chamfer_distance']:.8f}\n")
        f.write(f"cd_pred_to_gt:      {metrics['cd_pred_to_gt']:.8f}\n")
        f.write(f"cd_gt_to_pred:      {metrics['cd_gt_to_pred']:.8f}\n")
        f.write(f"hausdorff:          {metrics['hausdorff']:.8f}\n")
        f.write(f"f1:                 {metrics['f1']:.8f}\n")
        f.write(f"precision:          {metrics['precision']:.8f}\n")
        f.write(f"recall:             {metrics['recall']:.8f}\n")
        f.write(f"psnr_d1:            {metrics['psnr_d1']:.6f}\n")
    print(f"Metrics saved → {metrics_txt}")


if __name__ == "__main__":
    main()
