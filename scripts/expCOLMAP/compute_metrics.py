# scripts/expCOLMAP/compute_metrics.py
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_dust3r_inference import compute_all_metrics, load_ply, save_ply
from run_colmap import (
    co3d_camera_center,
    read_colmap_images_txt,
    umeyama,
    apply_similarity,
)
import gzip, json

# ── Paths ───────────────────────────────────────────────────────────────────
fused_ply = "outputs/colmap/dense_100fgpu/teddybear_101_11758_21048/dense/fused.ply"
gt_ply = "data/co3d/teddybear/101_11758_21048/pointcloud.ply"
sparse_dir = "outputs/colmap/dense_100fgpu/teddybear_101_11758_21048/sparse/0"
ann_file = "data/co3d/teddybear/frame_annotations.jgz"
seq_name = "101_11758_21048"

# ── Load COLMAP dense points ─────────────────────────────────────────────────
print(f"Loading COLMAP: {fused_ply}")
colmap_data = load_ply(fused_ply)
colmap_pts = colmap_data[0] if isinstance(colmap_data, tuple) else colmap_data
print(f"COLMAP points: {len(colmap_pts):,}")

# ── Load GT points ───────────────────────────────────────────────────────────
print(f"Loading GT: {gt_ply}")
gt_data = load_ply(gt_ply)
gt_pts = gt_data[0] if isinstance(gt_data, tuple) else gt_data
print(f"GT points: {len(gt_pts):,}")

# ── Load CO3D annotations for GT camera centers ──────────────────────────────
print("\n[ALIGNMENT] Loading CO3D annotations...")
with gzip.open(ann_file, "rt") as f:
    all_anns = json.load(f)
seq_anns = [a for a in all_anns if a["sequence_name"] == seq_name]
print(f"[ALIGNMENT] Found {len(seq_anns)} annotations")

# Build GT camera centers dict
co3d_centers = {}
for ann in seq_anns:
    fname = os.path.basename(ann["image"]["path"])
    co3d_centers[fname] = co3d_camera_center(
        ann["viewpoint"]["R"], ann["viewpoint"]["T"]
    )

# ── Load COLMAP camera centers ───────────────────────────────────────────────
img_txt = os.path.join(sparse_dir, "images.txt")
colmap_centers = read_colmap_images_txt(img_txt)
print(f"[ALIGNMENT] COLMAP cameras: {len(colmap_centers)}, GT cameras: {len(co3d_centers)}")

# ── Match cameras for Umeyama ────────────────────────────────────────────────
src_pts, dst_pts = [], []
for fname, center in colmap_centers.items():
    if fname in co3d_centers:
        src_pts.append(center)
        dst_pts.append(co3d_centers[fname])
matched = len(src_pts)
print(f"[ALIGNMENT] Matched cameras: {matched}")

# ── CORRECT ALIGNMENT: Use Umeyama directly (no pre-normalization) ──────────
if matched >= 3:
    src_arr = np.array(src_pts)
    dst_arr = np.array(dst_pts)
    
    # Compute scale from extents FIRST
    colmap_extent = (colmap_pts.max(axis=0) - colmap_pts.min(axis=0)).max()
    gt_extent = (gt_pts.max(axis=0) - gt_pts.min(axis=0)).max()
    extent_scale = gt_extent / colmap_extent
    
    print(f"\n[NORMALIZATION] COLMAP extent: {colmap_extent:.3f}")
    print(f"[NORMALIZATION] GT extent: {gt_extent:.3f}")
    print(f"[NORMALIZATION] Extent scale: {extent_scale:.6f}")
    
    # Umeyama on RAW camera centers (gets full similarity transform)
    umeyama_scale, R_align, t_align = umeyama(src_arr, dst_arr)
    print(f"[ALIGNMENT] Umeyama scale (cameras): {umeyama_scale:.4f}")
    
    # Use extent_scale for points, not Umeyama scale
    # Center COLMAP points
    colmap_center = colmap_pts.mean(axis=0)
    colmap_pts_centered = colmap_pts - colmap_center
    
    # Apply extent scale
    colmap_pts_scaled = colmap_pts_centered * extent_scale
    
    # Now apply rotation + translation from Umeyama (but recompute translation for scaled points)
    # Translation should align the centers
    gt_center = dst_arr.mean(axis=0)
    src_center = src_arr.mean(axis=0)
    
    # After scaling, the translation is:
    t_final = gt_center - (R_align @ (src_center * extent_scale))
    
    # Apply rotation and final translation
    colmap_pts_aligned = (R_align @ colmap_pts_scaled.T).T + t_final
    
    print(f"\n[DEBUG] Aligned COLMAP range:")
    print(f"  X: [{colmap_pts_aligned[:,0].min():.3f}, {colmap_pts_aligned[:,0].max():.3f}]")
    print(f"  Y: [{colmap_pts_aligned[:,1].min():.3f}, {colmap_pts_aligned[:,1].max():.3f}]")
    print(f"  Z: [{colmap_pts_aligned[:,2].min():.3f}, {colmap_pts_aligned[:,2].max():.3f}]")
    
    print(f"[DEBUG] GT range:")
    print(f"  X: [{gt_pts[:,0].min():.3f}, {gt_pts[:,0].max():.3f}]")
    print(f"  Y: [{gt_pts[:,1].min():.3f}, {gt_pts[:,1].max():.3f}]")
    print(f"  Z: [{gt_pts[:,2].min():.3f}, {gt_pts[:,2].max():.3f}]")
    
    final_pts = colmap_pts_aligned
else:
    print("[WARNING] Not enough matched cameras — skipping alignment!")
    final_pts = colmap_pts

# ── Subsample for faster metrics ─────────────────────────────────────────────
MAX_POINTS = 100000
if len(final_pts) > MAX_POINTS:
    print(f"\n[SUBSAMPLE] COLMAP: {len(final_pts):,} → {MAX_POINTS:,}")
    idx = np.random.choice(len(final_pts), MAX_POINTS, replace=False)
    final_pts = final_pts[idx]
if len(gt_pts) > MAX_POINTS:
    print(f"[SUBSAMPLE] GT: {len(gt_pts):,} → {MAX_POINTS:,}")
    idx = np.random.choice(len(gt_pts), MAX_POINTS, replace=False)
    gt_pts = gt_pts[idx]

# ── Compute metrics ──────────────────────────────────────────────────────────
print(f"\nComputing metrics on {len(final_pts):,} vs {len(gt_pts):,} points...")
metrics = compute_all_metrics(final_pts, gt_pts)

print(f"\n{'='*60}")
print(f"  Chamfer Distance: {metrics['chamfer_distance']:.6f}")
print(f"  cd_pred_to_gt:    {metrics['cd_pred_to_gt']:.6f}")
print(f"  cd_gt_to_pred:    {metrics['cd_gt_to_pred']:.6f}")
print(f"  F1 Score:         {metrics['f1']:.6f}")
print(f"{'='*60}\n")