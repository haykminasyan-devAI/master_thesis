"""
Run COLMAP SfM + MVS (Dense) on CO3D frames with GPU.

Usage:
    python scripts/expCOLMAP/run_colmap_dense_gpu.py \
        --masked_dir   data/co3d/teddybear/101_11758_21048/images \
        --sequence_dir data/co3d/teddybear/101_11758_21048 \
        --output_dir   outputs/colmap/dense_100f/teddybear_101_11758_21048 \
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
    parser.add_argument("--masked_dir",   required=True)
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--n_frames",     type=int, default=100)
    parser.add_argument("--gpu_index",    type=int, default=0)
    parser.add_argument("--skip_sfm",     action="store_true")
    parser.add_argument("--skip_dense",   action="store_true")
    args = parser.parse_args()

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

    # CO3D camera intrinsics
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
    print(f"[n={args.n_frames}] camera {W}×{H}  fx={fx:.1f} fy={fy:.1f}")

    selected = select_frames(images_abs, args.n_frames)
    actual_n = len(selected)
    print(f"[n={args.n_frames}] using {actual_n} frames")

    if actual_n < 3:
        print("WARNING: fewer than 3 frames — skipping.")
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

    # ── SPARSE SFM ────────────────────────────────────────────────────────────
    sparse_0 = os.path.join(sparse_dir, "0")
    if not args.skip_sfm:
        print("\n" + "="*60)
        print("  STEP 1-4: SPARSE SFM [GPU MODE]")
        print("="*60 + "\n")

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
        ], "Step 1 – Feature extraction [GPU]")

        run_cmd([
            "colmap", "sequential_matcher",
            "--database_path",              db_path,
            "--FeatureMatching.use_gpu",    "1",
            "--FeatureMatching.gpu_index",  str(args.gpu_index),
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.loop_detection", "0",
        ], "Step 2 – Sequential matching [GPU]")

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

        subdirs = sorted(d for d in os.listdir(sparse_dir)
                        if os.path.isdir(os.path.join(sparse_dir, d)))
        if not subdirs:
            print("No sub-model produced")
            return
        sparse_0 = os.path.join(sparse_dir, subdirs[0])

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

        run_cmd([
            "colmap", "image_undistorter",
            "--image_path",          images_abs,
            "--input_path",          sparse_0,
            "--output_path",         dense_dir,
            "--output_type",         "COLMAP",
            "--max_image_size",      "2000",
            "--min_scale",           "0.25",
        ], "Step 5 – Image undistortion")

        run_cmd([
            "colmap", "patch_match_stereo",
            "--workspace_path",      dense_dir,
            "--workspace_format",    "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
            "--PatchMatchStereo.use_gpu",          "1",
            "--PatchMatchStereo.gpu_index",        str(args.gpu_index),
            "--PatchMatchStereo.num_samples",      "15",
            "--PatchMatchStereo.num_iterations",   "10",
        ], "Step 6 – Patch match stereo [GPU]")

        run_cmd([
            "colmap", "stereo_fusion",
            "--workspace_path",      dense_dir,
            "--workspace_format",    "COLMAP",
            "--input_type",          "geometric",
            "--output_path",         fused_ply,
            "--StereoFusion.use_gpu", "1",
            "--StereoFusion.gpu_index", str(args.gpu_index),
            "--StereoFusion.min_num_pixels", "3",
        ], "Step 7 – Stereo fusion [GPU]")

    else:
        print(f"Skipping dense (--skip_dense); using {fused_ply}")

    if not os.path.isfile(fused_ply):
        print(f"ERROR: Dense fused.ply not found at {fused_ply}")
        return

    pred_pts_raw, pred_colors = load_ply(fused_ply)
    print(f"COLMAP dense: {len(pred_pts_raw):,} 3D points")
    save_ply(pred_pts_raw, pred_colors, raw_ply)

    gt_ply = os.path.join(args.sequence_dir, "pointcloud.ply")
    gt_pts = load_ply(gt_ply)
    print(f"GT: {len(gt_pts):,} points")

    img_txt = os.path.join(sparse_0, "images.txt")
    colmap_centers = read_colmap_images_txt(img_txt)

    src_pts, dst_pts = [], []
    for fname, center in colmap_centers.items():
        if fname in co3d_centers:
            src_pts.append(center)
            dst_pts.append(co3d_centers[fname])
    matched = len(src_pts)
    print(f"Camera correspondences: {matched}/{actual_n} registered")

    if matched < 3:
        shutil.copy(raw_ply, aligned_ply)
        pred_pts = pred_pts_raw
    else:
        colmap_extent = (pred_pts_raw.max(axis=0) - pred_pts_raw.min(axis=0)).max()
        gt_extent = (gt_pts.max(axis=0) - gt_pts.min(axis=0)).max()
        norm_scale = gt_extent / colmap_extent
        
        print(f"\n[NORMALIZATION] Scale factor: {norm_scale:.6f}")
        
        colmap_center = pred_pts_raw.mean(axis=0)
        pred_pts_norm = (pred_pts_raw - colmap_center) * norm_scale
        src_pts_norm = [(c - colmap_center) * norm_scale for c in colmap_centers.values()]
        
        scale_umeyama, R_align, t_align = umeyama(np.array(src_pts_norm), np.array(dst_pts))
        print(f"  Umeyama: scale={scale_umeyama:.4f}")
        
        pred_pts = apply_similarity(pred_pts_norm, 1.0, R_align, t_align)
        
        print(f"\n[DEBUG] Aligned COLMAP points range:")
        print(f"  X: [{pred_pts[:,0].min():.3f}, {pred_pts[:,0].max():.3f}]")
        print(f"  Y: [{pred_pts[:,1].min():.3f}, {pred_pts[:,1].max():.3f}]")
        print(f"  Z: [{pred_pts[:,2].min():.3f}, {pred_pts[:,2].max():.3f}]")
        
        save_ply(pred_pts, pred_colors, aligned_ply)

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