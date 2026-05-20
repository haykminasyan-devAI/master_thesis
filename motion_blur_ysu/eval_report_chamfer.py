#!/usr/bin/env python3
"""Chamfer table: pretrained-clean, pretrained-blur, LoRA-finetuned; val/test splits × 3 conditions."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import KDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
_stale_models = os.path.join(PROJECT_DIR, "motion_blur_ysu", "models")
if os.path.isfile(os.path.join(_stale_models, "lora_cross_attn.py")):
    print(
        "Removing stale motion_blur_ysu/models/ (conflicts with dust3r/croco `models`).",
        file=sys.stderr,
    )
    shutil.rmtree(_stale_models)

for p in (os.path.join(PROJECT_DIR, "dust3r"), PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401

from motion_blur_ysu.augmentation.motion_blur import DirectionalMotionBlur, apply_directional_blur_to_rgb
from motion_blur_ysu.dust3r_lora.lora_cross_attn import build_lora_dust3r_cross_attn

TEST_CATEGORIES = ["cup", "couch", "bottle", "teddybear", "donut", "toytrain"]
EVAL_CONDITIONS = ["clean-clean", "blur-blur", "clean-blur"]


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

    img_dir = str(img_proc) if img_proc.is_dir() else (str(img_raw) if img_raw.is_dir() else None)
    if img_dir is None:
        raise FileNotFoundError(f"Missing images for {category}/{seq_id}")

    gt_ply = str(gt_proc) if gt_proc.is_file() else (str(gt_raw) if gt_raw.is_file() else None)
    if gt_ply is None:
        raise FileNotFoundError(f"Missing pointcloud.ply for {category}/{seq_id}")
    return img_dir, gt_ply


def _blur_image_file(src_path: str, dst_path: str, rng: np.random.Generator):
    bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"read fail {src_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = apply_directional_blur_to_rgb(rgb, rng, DirectionalMotionBlur(3, 9))
    cv2.imwrite(dst_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def prepare_eval_images(img_paths, condition: str, tmp_root: str):
    """Return list of paths (original or blurred copies under tmp_root)."""
    if condition == "clean-clean":
        return list(img_paths)
    cfg = DirectionalMotionBlur(3, 9)
    out_paths = []
    for i, p in enumerate(img_paths):
        h = hashlib.md5(p.encode()).hexdigest()[:12]
        dst = os.path.join(tmp_root, f"f_{i}_{h}.jpg")
        if condition == "blur-blur":
            rng = np.random.default_rng(42 + i)
            _blur_image_file(p, dst, rng)
            out_paths.append(dst)
        elif condition == "clean-blur":
            # View A clean / view B blurred: alternate along the temporal stack (proxy for multi-view).
            if i % 2 == 0:
                shutil.copy2(p, dst)
            else:
                rng = np.random.default_rng(42 + i)
                _blur_image_file(p, dst, rng)
            out_paths.append(dst)
        else:
            raise ValueError(condition)
    return out_paths


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


def load_split_json(root, split_name):
    path = os.path.join(root, f"selected_seqs_{split_name}.json")
    with open(path, "r") as f:
        return json.load(f)


def iter_scenes(data, categories_filter):
    for cat, seqs in data.items():
        if categories_filter is not None and cat not in categories_filter:
            continue
        if not isinstance(seqs, dict):
            continue
        for sid in seqs:
            yield cat, sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d_processed", required=True)
    ap.add_argument("--co3d_raw", default=None)
    ap.add_argument("--split", default="test_10cat8seq")
    ap.add_argument("--categories", nargs="+", default=TEST_CATEGORIES)
    ap.add_argument("--dust3r_ckpt", required=True)
    ap.add_argument("--lora_ckpt", default=None)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_split_json(args.co3d_processed, args.split)
    scenes = list(iter_scenes(data, set(args.categories)))
    if not scenes:
        raise RuntimeError(f"No scenes for categories {args.categories} in split {args.split}")

    from dust3r.model import load_model

    def run_model(model, condition_name):
        cds = []
        tmp = tempfile.mkdtemp(prefix="mb_eval_")
        try:
            for cat, seq_id in scenes:
                img_dir, gt_ply = resolve_sequence_paths(
                    args.co3d_processed, cat, seq_id, args.co3d_raw
                )
                paths = frame_list(img_dir, args.n_frames)
                eval_paths = prepare_eval_images(paths, condition_name, tmp)
                cd = infer_sequence_cd(model, device, eval_paths, gt_ply, args.image_size)
                cds.append(cd)
                print(f"  {cat}/{seq_id} {condition_name} CD={cd:.8f}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return float(np.mean(cds)) if cds else float("nan")

    results = {"split": args.split, "scenes": len(scenes), "conditions": {}}

    for cond in EVAL_CONDITIONS:
        results["conditions"][cond] = {}

    # Pretrained
    base = load_model(args.dust3r_ckpt, device="cpu", verbose=False).to(device).eval()
    for cond in EVAL_CONDITIONS:
        results["conditions"][cond]["pretrained"] = run_model(base, cond)
    del base
    torch.cuda.empty_cache()

    # LoRA finetuned
    if args.lora_ckpt and os.path.isfile(args.lora_ckpt):
        lora = build_lora_dust3r_cross_attn(args.dust3r_ckpt, device="cpu").to(device).eval()
        try:
            ck = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(args.lora_ckpt, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        lora.load_state_dict(sd, strict=False)
        for cond in EVAL_CONDITIONS:
            results["conditions"][cond]["lora_finetuned"] = run_model(lora, cond)
        del lora
        torch.cuda.empty_cache()
    else:
        for cond in EVAL_CONDITIONS:
            results["conditions"][cond]["lora_finetuned"] = None

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Avg Chamfer (lower better) ===")
    print(f"{'condition':15s} {'pretrained':>14s} {'lora':>14s}")
    for cond in EVAL_CONDITIONS:
        a = results["conditions"][cond]["pretrained"]
        b = results["conditions"][cond].get("lora_finetuned")
        print(f"{cond:15s} {a:14.8f} {str(b) if b is None else f'{b:.8f}':>14s}")

    pc = results["conditions"]["clean-clean"]["pretrained"]
    pb = results["conditions"]["blur-blur"]["pretrained"]
    print("\n--- Summary (match paper-style rows) ---")
    print(f"  Pretrained DUSt3R, clean inputs (clean-clean):     {pc:.8f}")
    print(f"  Pretrained DUSt3R, blurred inputs (blur-blur):     {pb:.8f}")
    if results["conditions"]["blur-blur"].get("lora_finetuned") is not None:
        lb = results["conditions"]["blur-blur"]["lora_finetuned"]
        print(f"  LoRA-finetuned DUSt3R, blurred inputs (blur-blur): {lb:.8f}")


if __name__ == "__main__":
    main()
