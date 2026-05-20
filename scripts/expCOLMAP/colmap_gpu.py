"""
Run COLMAP SfM + MVS (Dense) on CO3D frames with GPU.
Optimized for SLURM + A100 GPUs.
Uses STD=1 normalization to match CO3D GT coordinate system.

Usage:
    python scripts/expCOLMAP/colmap_gpu.py \
        --masked_dir   data/co3d/teddybear/101_11758_21048/images \
        --sequence_dir data/co3d/teddybear/101_11758_21048 \
        --output_dir   outputs/colmap/dense_100fgpu/teddybear_101_11758_21048 \
        --n_frames     100 \
        --gpu_index    0
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked_dir",   required=True,
                        help="Directory containing pre-generated frames")
    parser.add_argument("--sequence_dir", required=True,
                        help="CO3D sequence directory (contains pointcloud.ply)")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to write COLMAP workspace and results")
    parser.add_argument("--n_frames",     type=int, default=100,
                        help="Number of evenly-spaced frames to feed into COLMAP")
    parser.add_argument("--gpu_index",    type=int, default=0,
                        help="GPU index for CUDA (default: 0)")
    parser.add_argument("--skip_sfm",     action="store_true",
                        help="Skip SfM if already done (just re-run dense + metrics)")
    parser.add_argument("--skip_dense",   action="store_true",
                        help="Skip dense reconstruction (just re-run metrics)")
    args = parser.parse_args()

    # ── Verify GPU is accessible ──────────────────────────────────────────────
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True)
        gpus = result.stdout.strip().split("\n")
        print(f"\n[GPU] Available GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
        print(f"[GPU] Using GPU index: {args.gpu_index}\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[WARNING] Could not verify GPU: {e}")
        print("[WARNING] COLMAP may fail if CUDA is not available\n")

    metrics_txt = os.path.join(args.output_dir, "metrics.txt")
    if os.path.isfile(metrics_txt):
        print(f"[skip] already done: {metrics_txt}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    db_path      = os.path.join(args.output_dir, "colmap.db")
    sparse_dir   = os.path.join(args.output_dir, "sparse")
    dense_dir    = os.path.join(args.output_dir, "dense")
    raw_ply      = os.path.join(args.output_dir, "predicted_raw.ply")
    aligned_ply  = os.path.join(args.output_dir, "predicted.ply")
    images_abs   = os.path.abspath(args.masked_dir)

    # ── CO3D camera intrinsics ────────────────────────────────────────────────
    seq_name = os.path.basename(args.sequence_dir)
    category = os.path.basename(os.path.dirname(args.sequence_dir))
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
    print(f"[n={args.n_frames}] camera {W}×{H}  fx={fx:.1f} fy={fy:.1f}")

    selected = select_frames(images_abs, args.n_frames)
    actual_n = len(selected)
    print(f"[n={args.n_frames}] using {actual_n} frames")

    if actual_n < 3:
        print("WARNING: fewer than 3 frames — skipping.")
        with open(metrics_txt, "w") as f:
            f.write(f"n_frames:  {args.n_frames}\nstatus:    skipped_too_few_frames\n")
        return

    img_list_path = os.path.join(args.output_dir, "image_list.txt")
    with open(img_list_path, "w") as f:
        f.write("\n".join(selected) + "\n")

    co3d_centers = {}
    for ann in annotations:
        fname = os.path.basename(ann["image"]["path"])
        co3d_centers[fname] = co3d_camera_center(
            ann["viewpoint"]["R"], ann["viewpoint"]["T"]
        )

    # ── SPARSE SFM [GPU] ──────────────────────────────────────────────────────
    sparse_0 = os.path.join(sparse_dir, "0")
    if not args.skip_sfm:
        print("\n" + "="*60)
        print("  STEP 1-4: SPARSE SFM [GPU MODE]")
        print("="*60 + "\n")

        # Step 1: Feature extraction (GPU SIFT)
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
            "--SiftExtraction.max_image_size",   "3200",
        ], "Step 1 – Feature extraction [GPU]")

        # Step 2: Sequential feature matching (GPU)
        run_cmd([
            "colmap", "sequential_matcher",
            "--database_path",              db_path,
            "--FeatureMatching.use_gpu",    "1",
            "--FeatureMatching.gpu_index",  str(args.gpu_index),
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.loop_detection", "0",
        ], "Step 2 – Sequential matching [GPU]")

        # Step 3: Sparse mapper (CPU-bound, but uses GPU-extracted features)
        os.makedirs(sparse_dir, exist_ok=True)
        run_cmd([
            "colmap", "mapper",
            "--database_path",                   db_path,
            "--image_path",                      images_abs,
            "--output_path",                     sparse_dir,
            "--Mapper.num_threads",              "8",
            "--Mapper.init_min_num_inliers",     "15",
            "--Mapper.abs_pose_min_num_inliers", "15",
        ], "Step 3 – Sparse mapper")

        # Pick the largest sub-model
        subdirs = sorted(d for d in os.listdir(sparse_dir)
                        if os.path.isdir(os.path.join(sparse_dir, d)))
        if not subdirs:
            print("No sub-model produced")
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
        ], "Step 4 – Model → TXT")

    else:
        print(f"Skipping SfM (--skip_sfm); using {sparse_0}")

    # ── DENSE MVS [GPU] ───────────────────────────────────────────────────────
    fused_ply = os.path.join(dense_dir, "fused.ply")
    if not args.skip_dense:
        print("\n" + "="*60)
        print("  STEP 5-7: DENSE RECONSTRUCTION [GPU MODE]")
        print("="*60 + "\n")

        # Step 5: Image Undistortion
        run_cmd([
            "colmap", "image_undistorter",
            "--image_path",          images_abs,
            "--input_path",          sparse_0,
            "--output_path",         dense_dir,
            "--output_type",         "COLMAP",
            "--max_image_size",      "2000",
            "--min_scale",           "0.25",
        ], "Step 5 – Image undistortion")

        # Step 6: Patch Match Stereo (GPU - auto-detects CUDA, no gpu_index flag)
        run_cmd([
            "colmap", "patch_match_stereo",
            "--workspace_path",      dense_dir,
            "--workspace_format",    "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
            "--PatchMatchStereo.num_samples",      "15",
            "--PatchMatchStereo.num_iterations",   "10",
            "--PatchMatchStereo.window_radius",    "5",
        ], "Step 6 – Patch match stereo [GPU]")

        # Step 7: Stereo Fusion (GPU - auto-detects CUDA, no gpu_index flag)
        run_cmd([
            "colmap", "stereo_fusion",
            "--workspace_path",      dense_dir,
            "--workspace_format",    "COLMAP",
            "--input_type",          "geometric",
            "--output_path",         fused_ply,
            "--StereoFusion.min_num_pixels", "3",
        ], "Step 7 – Stereo fusion [GPU]")

    else:
        print(f"Skipping dense (--skip_dense); using {fused_ply}")

    # ── Load dense point cloud ────────────────────────────────────────────────
    if not os.path.isfile(fused_ply):
        print(f"ERROR: Dense fused.ply not found at {fused_ply}")
        with open(metrics_txt, "w") as f:
            f.write(f"n_frames:  {args.n_frames}\nstatus:    no_dense_ply\n")
        return

    # Fix: Handle variable return from load_ply (may return 2 or 3 values)
    ply_data = load_ply(fused_ply)
    if isinstance(ply_data, tuple) and len(ply_data) >= 2:
        pred_pts_raw, pred_colors = ply_data[:2]
    else:
        pred_pts_raw = ply_data
        pred_colors = None
    
    print(f"COLMAP dense: {len(pred_pts_raw):,} 3D points")
    
    # Fix: Create default colors if None (save_ply doesn't handle None)
    if pred_colors is None:
        pred_colors = np.ones((len(pred_pts_raw), 3), dtype=np.uint8) * 128  # Gray
    
    save_ply(pred_pts_raw, pred_colors, raw_ply)

    # ── Load GT point cloud ───────────────────────────────────────────────────
    gt_ply = os.path.join(args.sequence_dir, "pointcloud.ply")
    gt_data = load_ply(gt_ply)
    if isinstance(gt_data, tuple) and len(gt_data) >= 2:
        gt_pts, gt_colors = gt_data[:2]
    else:
        gt_pts = gt_data
    print(f"GT: {len(gt_pts):,} points")

    # ── Read COLMAP camera centers for alignment ──────────────────────────────
    img_txt = os.path.join(sparse_0, "images.txt")
    colmap_centers = read_colmap_images_txt(img_txt)

    # ── Umeyama: align COLMAP → GT ────────────────────────────────────────────
    src_pts, dst_pts = [], []
    for fname, center in colmap_centers.items():
        if fname in co3d_centers:
            src_pts.append(center)
            dst_pts.append(co3d_centers[fname])
    matched = len(src_pts)
    print(f"Camera correspondences: {matched}/{actual_n} registered")

    if matched < 3:
        print(f"WARNING: only {matched} cameras registered — saving raw PLY.")
        shutil.copy(raw_ply, aligned_ply)
        pred_pts = pred_pts_raw
    else:
        # ── STD-BASED NORMALIZATION (matches CO3D GT) ────────────────────────
        # CO3D GT is normalized to have STD=1 across X,Y,Z axes
        # Compute average STD for COLMAP points
        colmap_std = np.std(pred_pts_raw, axis=0).mean()
        gt_std = np.std(gt_pts, axis=0).mean()
        
        # Scale COLMAP to match GT's STD=1 normalization
        std_scale = 1.0 / colmap_std  # Target STD=1 like CO3D GT
        
        print(f"\n[STD NORMALIZATION] COLMAP STD: {colmap_std:.4f}")
        print(f"[STD NORMALIZATION] GT STD: {gt_std:.4f}")
        print(f"[STD NORMALIZATION] Scale factor: {std_scale:.6f}")
        
        # Center and scale COLMAP points to STD=1
        colmap_center = pred_pts_raw.mean(axis=0)
        pred_pts_normalized = (pred_pts_raw - colmap_center) * std_scale
        
        # Also normalize camera centers the same way for Umeyama
        src_pts_normalized = [(c - colmap_center) * std_scale for c in src_pts]
        
        # Run Umeyama on normalized data (should get scale ≈ 1.0)
        scale_umeyama, R_align, t_align = umeyama(
            np.array(src_pts_normalized), np.array(dst_pts)
        )
        print(f"[ALIGNMENT] Umeyama scale: {scale_umeyama:.4f} (should be ~1.0)")
        
        # Apply rotation + translation (ignore Umeyama scale since we already normalized)
        pred_pts = apply_similarity(pred_pts_normalized, 1.0, R_align, t_align)
        
        print(f"\n[DEBUG] Aligned COLMAP points range:")
        print(f"  X: [{pred_pts[:,0].min():.3f}, {pred_pts[:,0].max():.3f}]")
        print(f"  Y: [{pred_pts[:,1].min():.3f}, {pred_pts[:,1].max():.3f}]")
        print(f"  Z: [{pred_pts[:,2].min():.3f}, {pred_pts[:,2].max():.3f}]")
        print(f"[DEBUG] GT points range:")
        print(f"  X: [{gt_pts[:,0].min():.3f}, {gt_pts[:,0].max():.3f}]")
        print(f"  Y: [{gt_pts[:,1].min():.3f}, {gt_pts[:,1].max():.3f}]")
        print(f"  Z: [{gt_pts[:,2].min():.3f}, {gt_pts[:,2].max():.3f}]")
        print(f"{'='*60}\n")
        
        save_ply(pred_pts, pred_colors, aligned_ply)
        print(f"Saved aligned PLY → {aligned_ply}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_all_metrics(pred_pts, gt_pts)

    print(f"\n{'='*60}")
    print(f"  COLMAP DENSE [GPU]  n_frames={args.n_frames}")
    print(f"  Chamfer={metrics['chamfer_distance']:.6f}")
    print(f"  cd_pred_to_gt={metrics['cd_pred_to_gt']:.6f}")
    print(f"  cd_gt_to_pred={metrics['cd_gt_to_pred']:.6f}")
    print(f"{'='*60}\n")

    with open(metrics_txt, "w") as f:
        f.write(f"method:             COLMAP_DENSE_GPU\n")
        f.write(f"n_frames:           {args.n_frames}\n")
        f.write(f"cameras_registered: {matched}\n")
        f.write(f"n_pred_points:      {metrics['n_pred_points']}\n")
        f.write(f"chamfer_distance:   {metrics['chamfer_distance']:.8f}\n")
        f.write(f"cd_pred_to_gt:      {metrics['cd_pred_to_gt']:.8f}\n")
        f.write(f"cd_gt_to_pred:      {metrics['cd_gt_to_pred']:.8f}\n")
        f.write(f"status:             ok\n")
    print(f"Metrics saved → {metrics_txt}")


if __name__ == "__main__":
    main()