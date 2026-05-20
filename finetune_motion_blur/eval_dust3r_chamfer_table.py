#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
for p in [os.path.join(PROJECT_DIR, "dust3r"), PROJECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_ply_xyz(path):
    import trimesh

    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices if hasattr(mesh, "vertices") else mesh.points, dtype=np.float32)


def compute_chamfer(pred, gt):
    tree_gt = KDTree(gt)
    tree_pred = KDTree(pred)
    d_p2g, _ = tree_gt.query(pred)
    d_g2p, _ = tree_pred.query(gt)
    cd_p2g = float(np.mean(d_p2g ** 2))
    cd_g2p = float(np.mean(d_g2p ** 2))
    return 0.5 * (cd_p2g + cd_g2p)


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


def resolve_sequence_paths(co3d_root, category, seq_id, co3d_raw_root=None):
    root = Path(co3d_root)
    seq_proc = root / category / seq_id
    img_proc = seq_proc / "images"
    gt_proc = seq_proc / "pointcloud.ply"

    raw = Path(co3d_raw_root) if co3d_raw_root else root.parent / "co3d"
    img_raw = raw / category / seq_id / "images"
    gt_raw = raw / category / seq_id / "pointcloud.ply"

    img_dir = None
    if img_proc.is_dir():
        img_dir = str(img_proc)
    elif img_raw.is_dir():
        img_dir = str(img_raw)
    if img_dir is None:
        raise FileNotFoundError(f"Missing images for {category}/{seq_id}")

    gt_ply = None
    if gt_proc.is_file():
        gt_ply = str(gt_proc)
    elif gt_raw.is_file():
        gt_ply = str(gt_raw)
    if gt_ply is None:
        raise FileNotFoundError(f"Missing pointcloud.ply for {category}/{seq_id}")

    return img_dir, gt_ply


def infer_sequence_cd(model, device, img_paths, gt_ply, image_size):
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    imgs = load_images(img_paths, size=image_size)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=300, schedule="linear", lr=0.01)

    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())
    pred_pts = np.concatenate([pts3d[i][masks[i]] for i in range(len(pts3d))], axis=0).astype(np.float32)
    gt_pts = load_ply_xyz(gt_ply)
    return compute_chamfer(pred_pts, gt_pts)


def load_split_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seq_ids_for_category(split_data, category):
    v = split_data.get(category, {})
    if isinstance(v, dict):
        return sorted(v.keys())
    if isinstance(v, list):
        return sorted(v)
    return []


def evaluate_checkpoint(args, model, categories, split_data, label):
    rows = []
    for cat in categories:
        seqs = seq_ids_for_category(split_data, cat)
        if args.max_seqs_per_category > 0:
            seqs = seqs[: args.max_seqs_per_category]
        for seq_id in seqs:
            img_dir, gt_ply = resolve_sequence_paths(args.co3d_root, cat, seq_id, args.co3d_raw_root)
            paths = frame_list(img_dir, args.n_frames)
            cd = infer_sequence_cd(model, args.device, paths, gt_ply, args.image_size)
            rows.append({"model": label, "category": cat, "seq_id": seq_id, "chamfer": float(cd)})
            print(f"[{label}] {cat}/{seq_id} chamfer={cd:.8f}")
    return rows


def summarize(rows, categories):
    out = {}
    for c in categories:
        vals = [r["chamfer"] for r in rows if r["category"] == c]
        out[c] = float(np.mean(vals)) if vals else None
    all_vals = [r["chamfer"] for r in rows]
    out["AVG_ALL"] = float(np.mean(all_vals)) if all_vals else None
    return out


def print_table(sum_pre, sum_ft, categories):
    headers = ["Category", "Pretrained_DUSt3R_CD", "Finetuned_DUSt3R_CD"]
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for c in categories + ["AVG_ALL"]:
        a = sum_pre.get(c)
        b = sum_ft.get(c)
        sa = f"{a:.8f}" if a is not None else "NA"
        sb = f"{b:.8f}" if b is not None else "NA"
        print(f"{c} | {sa} | {sb}")


def main():
    ap = argparse.ArgumentParser("Evaluate averaged Chamfer: pretrained vs finetuned DUSt3R")
    ap.add_argument("--co3d_root", required=True)
    ap.add_argument("--co3d_raw_root", default=None)
    ap.add_argument("--split_json", required=True, help="JSON mapping category -> sequence ids")
    ap.add_argument(
        "--categories",
        nargs="+",
        default=["cup", "couch", "bottle", "teddybear", "donut", "toytrain"],
    )
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--finetuned_ckpt", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--max_seqs_per_category", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from dust3r.model import load_model

    split_data = load_split_json(args.split_json)
    categories = [c for c in args.categories if len(seq_ids_for_category(split_data, c)) > 0]
    if not categories:
        raise RuntimeError("None of requested categories have sequences in split_json.")

    model_pre = load_model(args.pretrained_ckpt, device="cpu", verbose=False).to(args.device).eval()
    rows_pre = evaluate_checkpoint(args, model_pre, categories, split_data, "pretrained")
    del model_pre
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_ft = load_model(args.finetuned_ckpt, device="cpu", verbose=False).to(args.device).eval()
    rows_ft = evaluate_checkpoint(args, model_ft, categories, split_data, "finetuned")

    sum_pre = summarize(rows_pre, categories)
    sum_ft = summarize(rows_ft, categories)

    result = {
        "categories": categories,
        "summary": {"pretrained": sum_pre, "finetuned": sum_ft},
        "rows_pretrained": rows_pre,
        "rows_finetuned": rows_ft,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print_table(sum_pre, sum_ft, categories)
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
