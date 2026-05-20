"""
Run COLMAP Structure-from-Motion on CO3D masked frames and compare
the reconstructed point cloud against the CO3D GT using the same
metrics as DUSt3R (Chamfer, Hausdorff, F1, PSNR d1).

The COLMAP sparse reconstruction lives in an arbitrary coordinate frame,
so we align it to the CO3D GT frame using the camera centers: we compute
the optimal similarity transform (scale, R, t) that maps COLMAP camera
positions to the known CO3D camera positions, then apply it to the
sparse 3D points before computing metrics.

Usage:
    python scripts/run_colmap.py \\
        --images_dir   outputs/dust3r/exp2images/20 \\
        --sequence_dir data/co3d/teddybear/101_11758_21048 \\
        --output_dir   outputs/colmap/frames_20 \\
        --n_frames     20 \\
        --mask_ratio   0.25 \\
        --gpu_index    0
"""

import argparse
import gzip
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_dust3r_inference import compute_all_metrics, load_ply, save_ply


# ── CO3D helpers ──────────────────────────────────────────────────────────────

def read_co3d_annotations(annotation_file: str, sequence_name: str) -> list:
    with gzip.open(annotation_file, "rt") as f:
        data = json.load(f)
    items = [a for a in data if a["sequence_name"] == sequence_name]
    items.sort(key=lambda x: x["frame_number"])
    return items


def ndc_to_pixel_intrinsics(focal_length, principal_point, image_hw):
    """Convert CO3D ndc_isotropic intrinsics to pixel-space (COLMAP PINHOLE)."""
    H, W = image_hw
    s = min(H, W) / 2.0
    fx = focal_length[0] * s
    fy = focal_length[1] * s
    cx = (W - 1) / 2.0 - principal_point[0] * s
    cy = (H - 1) / 2.0 - principal_point[1] * s
    return fx, fy, cx, cy


def co3d_camera_center(R_list, T_list) -> np.ndarray:
    """World-space camera center from CO3D extrinsics: C = -R^T @ T."""
    R = np.array(R_list).reshape(3, 3)
    T = np.array(T_list).reshape(3)
    return -R.T @ T


# ── COLMAP binary reader ───────────────────────────────────────────────────────

def read_colmap_points3d_bin(path: str):
    pts, colors = [], []
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            struct.unpack("<Q", f.read(8))          # point3d_id
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            struct.unpack("<d", f.read(8))           # reproj error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)                    # skip track
            pts.append(xyz)
            colors.append(rgb)
    return np.array(pts, dtype=np.float32), np.array(colors, dtype=np.float32)


def read_colmap_points3d_txt(path: str):
    pts, colors = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
    return np.array(pts, dtype=np.float32), np.array(colors, dtype=np.float32)


def read_colmap_images_bin(path: str) -> dict:
    """Return {image_name: camera_center_xyz} from images.bin."""
    images = {}
    with open(path, "rb") as f:
        num_reg = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg):
            struct.unpack("<I", f.read(4))              # image_id
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz     = struct.unpack("<3d", f.read(24))
            struct.unpack("<I", f.read(4))              # camera_id
            # read null-terminated name
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * 24)                      # skip 2D points

            # convert quaternion to rotation matrix
            R = _quat_to_R(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            center = -R.T @ t                           # world-space camera center
            images[name.decode()] = center
    return images


def read_colmap_images_txt(path: str) -> dict:
    images = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz     = map(float, parts[5:8])
        name = parts[9]
        R = _quat_to_R(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        images[name] = -R.T @ t
        i += 2                                          # skip 2D-point line
    return images


def _quat_to_R(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) to 3×3 rotation matrix."""
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy+qz*qz),   s*(qx*qy - qz*qw),   s*(qx*qz + qy*qw)],
        [  s*(qx*qy + qz*qw), 1 - s*(qx*qx+qz*qz),   s*(qy*qz - qx*qw)],
        [  s*(qx*qz - qy*qw),   s*(qy*qz + qx*qw), 1 - s*(qx*qx+qy*qy)],
    ])


# ── Umeyama similarity transform ─────────────────────────────────────────────

def umeyama(src: np.ndarray, dst: np.ndarray):
    """
    Compute optimal similarity transform dst ≈ scale * R @ src + t.
    src, dst: (N, 3) with N≥3 corresponding points.
    Returns (scale, R[3x3], t[3]).
    """
    n = len(src)
    mu_s = src.mean(0);  mu_d = dst.mean(0)
    src_c = src - mu_s;  dst_c = dst - mu_d
    var_s = float((src_c ** 2).sum()) / n
    cov   = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0] * (len(S) - 1) + [d])
    R     = U @ D @ Vt
    scale = float((S * D.diagonal()).sum()) / var_s
    t     = mu_d - scale * R @ mu_s
    return scale, R, t


def apply_similarity(pts: np.ndarray, scale: float,
                     R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ pts.T)).T + t


# ── subprocess helper ─────────────────────────────────────────────────────────

def run_cmd(cmd: list, label: str = ""):
    print(f"\n{'─'*65}")
    print(f"  {label or cmd[0]}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*65}")
    subprocess.run(cmd, check=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir",   required=True,
                        help="Folder with input images (masked or clean)")
    parser.add_argument("--sequence_dir", required=True,
                        help="CO3D sequence directory (contains pointcloud.ply)")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to write COLMAP workspace and results")
    parser.add_argument("--n_frames",     type=int,   default=20)
    parser.add_argument("--mask_ratio",   type=float, default=0.25)
    parser.add_argument("--gpu_index",    type=int,   default=0,
                        help="GPU index for SIFT extraction/matching (default: 0)")
    parser.add_argument("--skip_sfm",     action="store_true",
                        help="Skip feature extraction/matching/mapper if already done")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    db_path     = os.path.join(args.output_dir, "colmap.db")
    sparse_dir  = os.path.join(args.output_dir, "sparse")
    raw_ply     = os.path.join(args.output_dir, "predicted_raw.ply")
    aligned_ply = os.path.join(args.output_dir, "predicted.ply")
    metrics_txt = os.path.join(args.output_dir, "metrics.txt")
    images_abs  = os.path.abspath(args.images_dir)

    # ── load CO3D intrinsics for the sequence ─────────────────────────────────
    seq_name = os.path.basename(args.sequence_dir)
    cat_dir  = os.path.dirname(args.sequence_dir)
    ann_file = os.path.join(cat_dir, "frame_annotations.jgz")
    print(f"Loading CO3D annotations from {ann_file} ...")
    annotations = read_co3d_annotations(ann_file, seq_name)
    ann0 = annotations[0]
    H, W = ann0["image"]["size"]
    fx, fy, cx, cy = ndc_to_pixel_intrinsics(
        ann0["viewpoint"]["focal_length"],
        ann0["viewpoint"]["principal_point"],
        (H, W),
    )
    print(f"Camera (PINHOLE): {W}×{H}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # ── list input images ──────────────────────────────────────────────────────
    img_files = sorted(f for f in os.listdir(images_abs)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"\nInput: {len(img_files)} frames from {images_abs}")

    # ── CO3D GT camera centers (keyed by frame filename) ──────────────────────
    co3d_centers = {}
    for ann in annotations:
        fname = os.path.basename(ann["image"]["path"])
        co3d_centers[fname] = co3d_camera_center(
            ann["viewpoint"]["R"], ann["viewpoint"]["T"]
        )

    if not args.skip_sfm:
        # ── feature extraction (GPU SIFT, fixed PINHOLE intrinsics) ──────────
        run_cmd([
            "colmap", "feature_extractor",
            "--database_path",                  db_path,
            "--image_path",                     images_abs,
            "--ImageReader.single_camera",      "1",
            "--ImageReader.camera_model",       "PINHOLE",
            "--ImageReader.camera_params",      f"{fx:.6f},{fy:.6f},{cx:.6f},{cy:.6f}",
            "--SiftExtraction.use_gpu",         "1",
            "--SiftExtraction.gpu_index",       str(args.gpu_index),
            "--SiftExtraction.max_num_features","8192",
        ], "Step 1 – Feature extraction (GPU SIFT, fixed intrinsics)")

        # ── sequential feature matching (GPU) ─────────────────────────────────
        run_cmd([
            "colmap", "sequential_matcher",
            "--database_path",                              db_path,
            "--SiftMatching.use_gpu",                       "1",
            "--SiftMatching.gpu_index",                     str(args.gpu_index),
            "--SequentialMatching.overlap",                 "15",
            "--SequentialMatching.loop_detection",          "1",
            "--SequentialMatching.loop_detection_num_images","30",
        ], "Step 2 – Sequential feature matching (GPU)")

        # ── sparse mapper (SfM) ───────────────────────────────────────────────
        os.makedirs(sparse_dir, exist_ok=True)
        run_cmd([
            "colmap", "mapper",
            "--database_path",                      db_path,
            "--image_path",                         images_abs,
            "--output_path",                        sparse_dir,
            "--Mapper.num_threads",                 "8",
            "--Mapper.init_min_num_inliers",        "15",
            "--Mapper.abs_pose_min_num_inliers",    "15",
            "--Mapper.ba_global_images_ratio",      "1.1",
            "--Mapper.ba_global_points_ratio",      "1.1",
        ], "Step 3 – Sparse mapper (SfM)")

        # ── convert binary model to TXT for easy inspection ───────────────────
        sparse_0 = os.path.join(sparse_dir, "0")
        if not os.path.isdir(sparse_0):
            # check if there are other submodels
            subdirs = [d for d in os.listdir(sparse_dir)
                       if os.path.isdir(os.path.join(sparse_dir, d))]
            if not subdirs:
                raise RuntimeError("COLMAP mapper produced no model. "
                                   "Try with more images or looser thresholds.")
            sparse_0 = os.path.join(sparse_dir, sorted(subdirs)[0])
            print(f"Using sparse sub-model: {sparse_0}")

        run_cmd([
            "colmap", "model_converter",
            "--input_path",  sparse_0,
            "--output_path", sparse_0,
            "--output_type", "TXT",
        ], "Step 4 – Convert model to TXT")
    else:
        sparse_0 = os.path.join(sparse_dir, "0")
        print(f"Skipping SfM (--skip_sfm); using model at {sparse_0}")

    # ── read COLMAP 3D points ─────────────────────────────────────────────────
    p3d_txt = os.path.join(sparse_0, "points3D.txt")
    p3d_bin = os.path.join(sparse_0, "points3D.bin")
    if os.path.isfile(p3d_txt):
        pred_pts_raw, pred_colors = read_colmap_points3d_txt(p3d_txt)
    elif os.path.isfile(p3d_bin):
        pred_pts_raw, pred_colors = read_colmap_points3d_bin(p3d_bin)
    else:
        raise FileNotFoundError(f"No points3D file in {sparse_0}")
    print(f"\nCOLMAP sparse: {len(pred_pts_raw):,} 3D points")
    save_ply(pred_pts_raw, pred_colors, raw_ply)

    # ── read COLMAP estimated camera centers ──────────────────────────────────
    img_txt = os.path.join(sparse_0, "images.txt")
    img_bin = os.path.join(sparse_0, "images.bin")
    if os.path.isfile(img_txt):
        colmap_centers = read_colmap_images_txt(img_txt)
    elif os.path.isfile(img_bin):
        colmap_centers = read_colmap_images_bin(img_bin)
    else:
        raise FileNotFoundError(f"No images file in {sparse_0}")

    # ── build matched camera-center pairs for Umeyama alignment ──────────────
    src_pts, dst_pts = [], []
    matched = 0
    for fname in colmap_centers:
        if fname in co3d_centers:
            src_pts.append(colmap_centers[fname])    # COLMAP frame
            dst_pts.append(co3d_centers[fname])      # CO3D / GT frame
            matched += 1

    print(f"\nCamera-center correspondences: {matched} / {len(img_files)} frames registered")

    if matched < 3:
        print("WARNING: fewer than 3 cameras registered – cannot compute similarity "
              "transform. Saving raw (unaligned) point cloud as predicted.ply.")
        shutil.copy(raw_ply, aligned_ply)
        pred_pts = pred_pts_raw
    else:
        src_arr = np.array(src_pts)
        dst_arr = np.array(dst_pts)
        scale, R_align, t_align = umeyama(src_arr, dst_arr)
        print(f"  Umeyama: scale={scale:.4f}")

        pred_pts = apply_similarity(pred_pts_raw, scale, R_align, t_align)
        save_ply(pred_pts, pred_colors, aligned_ply)
        print(f"Saved aligned PLY → {aligned_ply}")

    # ── load GT point cloud ───────────────────────────────────────────────────
    gt_ply_path = os.path.join(args.sequence_dir, "pointcloud.ply")
    gt_pts = load_ply(gt_ply_path)
    print(f"GT: {len(gt_pts):,} points")

    # ── compute all metrics ───────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics = compute_all_metrics(pred_pts, gt_pts)

    print("\n" + "=" * 65)
    print("  COLMAP RESULTS")
    print("=" * 65)
    print(f"  n_frames:             {args.n_frames}  (mask_ratio={args.mask_ratio})")
    print(f"  COLMAP points:        {metrics['n_pred_points']:,}")
    print(f"  GT points:            {metrics['n_gt_points']:,}")
    print(f"  Chamfer Distance:     {metrics['chamfer_distance']:.6f}")
    print(f"  CD pred→GT:           {metrics['cd_pred_to_gt']:.6f}")
    print(f"  CD GT→pred:           {metrics['cd_gt_to_pred']:.6f}")
    print(f"  Hausdorff:            {metrics['hausdorff']:.6f}")
    print(f"  F1 (1% bbox diag):    {metrics['f1']:.6f}")
    print(f"  Precision:            {metrics['precision']:.6f}")
    print(f"  Recall:               {metrics['recall']:.6f}")
    print(f"  PSNR d1:              {metrics['psnr_d1']:.4f} dB")
    print("=" * 65)

    with open(metrics_txt, "w") as f:
        f.write(f"method:            COLMAP\n")
        f.write(f"sequence:          {args.sequence_dir}\n")
        f.write(f"n_frames:          {args.n_frames}\n")
        f.write(f"mask_ratio:        {args.mask_ratio}\n")
        f.write(f"cameras_registered:{matched}\n")
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
    print(f"Metrics saved → {metrics_txt}")


if __name__ == "__main__":
    main()
