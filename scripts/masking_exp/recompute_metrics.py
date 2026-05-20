"""
Recompute all metrics from already-saved predicted.ply files.
Use this to add new metrics (Hausdorff, F1, PSNR d1) to existing exp outputs
without re-running DUSt3R inference.

Usage:
    python scripts/recompute_metrics.py \
        --exp_dir   outputs/dust3r/exp4 \
        --gt_ply    data/co3d/teddybear/101_11758_21048/pointcloud.ply

    # Or for a single run directory:
    python scripts/recompute_metrics.py \
        --exp_dir   outputs/dust3r/exp4/frames_10 \
        --gt_ply    data/co3d/teddybear/101_11758_21048/pointcloud.ply \
        --single
"""

import os
import sys
import argparse
import numpy as np

# reuse metric functions from run_dust3r_inference
sys.path.insert(0, os.path.dirname(__file__))
from run_dust3r_inference import compute_all_metrics, load_ply, save_ply


def recompute_one(run_dir: str, gt_pts: np.ndarray) -> bool:
    pred_ply = os.path.join(run_dir, "predicted.ply")
    metrics_path = os.path.join(run_dir, "metrics.txt")

    if not os.path.isfile(pred_ply):
        print(f"  [skip] no predicted.ply in {run_dir}")
        return False

    # read existing metrics.txt to preserve run metadata
    meta = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()

    pred_pts = load_ply(pred_ply)
    metrics  = compute_all_metrics(pred_pts, gt_pts)

    with open(metrics_path, "w") as f:
        # preserve original metadata fields
        for key in ("sequence", "n_frames", "n_masked", "n_clean",
                    "mask_ratio", "n_pred_points", "n_gt_points"):
            if key in meta:
                f.write(f"{key}:          {meta[key]}\n")

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

    print(f"  [OK] n={meta.get('n_frames','?'):>3}  "
          f"CD={metrics['chamfer_distance']:.4f}  "
          f"HD={metrics['hausdorff']:.4f}  "
          f"F1={metrics['f1']:.4f}  "
          f"PSNR_d1={metrics['psnr_d1']:.2f}dB")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_dir', required=True,
                        help='Experiment output directory (e.g. outputs/dust3r/exp4)')
    parser.add_argument('--gt_ply',  required=True,
                        help='Path to GT pointcloud.ply')
    parser.add_argument('--single',  action='store_true',
                        help='Treat exp_dir as a single run directory, not a sweep')
    args = parser.parse_args()

    print(f"Loading GT point cloud from {args.gt_ply} ...")
    gt_pts = load_ply(args.gt_ply)
    print(f"GT: {len(gt_pts):,} points\n")

    if args.single:
        recompute_one(args.exp_dir, gt_pts)
        return

    # find all frames_XX subdirectories
    subdirs = sorted([
        os.path.join(args.exp_dir, d)
        for d in os.listdir(args.exp_dir)
        if d.startswith("frames_") and os.path.isdir(os.path.join(args.exp_dir, d))
    ])

    if not subdirs:
        print("No frames_XX subdirectories found. Use --single for a single run dir.")
        return

    print(f"Found {len(subdirs)} run directories in {args.exp_dir}\n")
    ok = 0
    for d in subdirs:
        if recompute_one(d, gt_pts):
            ok += 1

    print(f"\nDone. Recomputed metrics for {ok}/{len(subdirs)} runs.")


if __name__ == '__main__':
    main()
