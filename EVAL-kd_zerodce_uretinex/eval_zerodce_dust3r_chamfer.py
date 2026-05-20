#!/usr/bin/env python3
"""
DUSt3R Chamfer evaluation with Zero-DCE (KD) student vs clean/dark CO3D images.

Scenarios (same DUSt3R weights for all):
  - dust3r_clean: original RGB frames
  - dust3r_dark: synthetic low-light (matches KD-Zero-Reference val-style brightness/gamma + noise)
  - zerodce_dust3r_dark: low-light -> Zero-DCE -> DUSt3R
  - zerodce_dust3r_clean: clean -> Zero-DCE -> DUSt3R
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import KDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
KDZR_DIR = os.path.join(PROJECT_DIR, "KD-Zero-Reference")
for p in (os.path.join(PROJECT_DIR, "dust3r"), PROJECT_DIR, KDZR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.model import load_model as load_dust3r_model

from synthetic_lowlight import create_synthetic_low_light  # noqa: E402

DEFAULT_CATEGORIES = [
    "bottle",
    "couch",
    "cup",
    "donut",
    "hydrant",
    "teddybear",
    "toybus",
    "toytrain",
]


def load_ply_xyz(path):
    import trimesh

    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices if hasattr(mesh, "vertices") else mesh.points, dtype=np.float32)


def compute_chamfer(pred, gt):
    tree_gt = KDTree(gt)
    tree_pred = KDTree(pred)
    d_p2g, _ = tree_gt.query(pred)
    d_g2p, _ = tree_pred.query(gt)
    cd_p2g = float(np.mean(d_p2g**2))
    cd_g2p = float(np.mean(d_g2p**2))
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


def load_split_json(root, split_name):
    candidates = [Path(root) / f"selected_seqs_{split_name}.json"]
    if "_" in split_name:
        candidates.append(Path(root) / f"selected_seqs_{split_name.split('_', 1)[0]}.json")
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    looked = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing split json. Looked for:\n - {looked}")


def iter_scenes(data, categories_filter):
    for cat, seqs in data.items():
        if categories_filter is not None and cat not in categories_filter:
            continue
        if not isinstance(seqs, dict):
            continue
        for sid in sorted(seqs.keys()):
            yield cat, sid


def _frame_rng(cat: str, seq_id: str, frame_idx: int, base_seed: int):
    h = zlib.adler32(f"{cat}/{seq_id}/{frame_idx}".encode()) & 0xFFFFFFFF
    return np.random.default_rng(int(h ^ (base_seed & 0xFFFFFFFF)))


def synthetic_dark_bgr(
    bgr: np.ndarray,
    cat: str,
    seq_id: str,
    frame_idx: int,
    brightness: float,
    gamma: float,
    noise_seed: int,
) -> np.ndarray:
    rng = _frame_rng(cat, seq_id, frame_idx, noise_seed)
    return create_synthetic_low_light(
        bgr,
        brightness_factor=float(brightness),
        gamma=float(gamma),
        rng=rng,
    )


def load_zerodce(zerodce_root: str, student_ckpt: str, device: torch.device):
    from student_zerodce import load_zerodce_student

    net = load_zerodce_student(zerodce_root).to(device).eval()
    try:
        ck = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(student_ckpt, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    for p in net.parameters():
        p.requires_grad = False
    return net


def run_zerodce_on_bgr(net, device: torch.device, bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        _, enh, _ = net(x)
    enh = enh.clamp(0.0, 1.0)
    out_rgb = enh[0].permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((out_rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def build_frame_paths(
    base_paths: list[str],
    scenario: str,
    zerodce_net,
    device: torch.device,
    tmp_dir: str,
    cat: str,
    seq_id: str,
    brightness: float,
    gamma: float,
    noise_seed: int,
) -> list[str]:
    out_paths = []
    for i, p in enumerate(base_paths):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")

        if scenario == "dust3r_clean":
            proc = bgr
        elif scenario == "dust3r_dark":
            proc = synthetic_dark_bgr(bgr, cat, seq_id, i, brightness, gamma, noise_seed)
        elif scenario == "zerodce_dust3r_dark":
            dark = synthetic_dark_bgr(bgr, cat, seq_id, i, brightness, gamma, noise_seed)
            proc = run_zerodce_on_bgr(zerodce_net, device, dark)
        elif scenario == "zerodce_dust3r_clean":
            proc = run_zerodce_on_bgr(zerodce_net, device, bgr)
        else:
            raise ValueError(scenario)

        dst = os.path.join(tmp_dir, f"f_{i:03d}.jpg")
        cv2.imwrite(dst, proc)
        out_paths.append(dst)
    return out_paths


def infer_sequence_cd(model, device, img_paths, gt_ply, image_size, align_niter):
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
        scene.compute_global_alignment(init="mst", niter=int(align_niter), schedule="linear", lr=0.01)
    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())
    pred_pts = np.concatenate([pts3d[i][masks[i]] for i in range(len(pts3d))], axis=0).astype(np.float32)
    gt_pts = load_ply_xyz(gt_ply)
    return compute_chamfer(pred_pts, gt_pts)


def summarize_rows(rows):
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["chamfer"])
    mean_by_cat = {k: float(np.mean(v)) for k, v in by_cat.items()}
    mean_all = float(np.mean([r["chamfer"] for r in rows])) if rows else float("nan")
    return mean_by_cat, mean_all


def main():
    ap = argparse.ArgumentParser("Zero-DCE (KD) + DUSt3R Chamfer on CO3D")
    ap.add_argument("--co3d_processed", required=True)
    ap.add_argument("--co3d_raw", default=None)
    ap.add_argument("--split", default="test", help="e.g. test -> selected_seqs_test.json")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--dust3r_ckpt", required=True)
    ap.add_argument("--student_ckpt", required=True, help="e.g. student_best_val.pth")
    ap.add_argument("--zerodce_root", default=os.path.join(PROJECT_DIR, "external", "Zero-DCE"))
    ap.add_argument("--brightness", type=float, default=0.12, help="Synthetic low-light (val-style default)")
    ap.add_argument("--gamma", type=float, default=2.2)
    ap.add_argument("--noise_seed", type=int, default=42, help="XOR seed for per-frame noise RNG")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--align_niter", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_data = load_split_json(args.co3d_processed, args.split)
    scenes = list(iter_scenes(split_data, set(args.categories)))
    if not scenes:
        raise RuntimeError(f"No scenes for categories {args.categories} in split {args.split}")

    print(f"[startup] device={device} scenes={len(scenes)} n_frames={args.n_frames}", flush=True)
    print(f"[startup] categories={args.categories}", flush=True)
    print(f"[startup] synthetic low-light brightness={args.brightness} gamma={args.gamma}", flush=True)

    dust3r = load_dust3r_model(args.dust3r_ckpt, device="cpu", verbose=False).to(device).eval()
    zerodce = load_zerodce(args.zerodce_root, args.student_ckpt, device)

    scenarios = [
        "dust3r_clean",
        "dust3r_dark",
        "zerodce_dust3r_dark",
        "zerodce_dust3r_clean",
    ]

    all_rows = []
    summary = {}
    tmp_root = tempfile.mkdtemp(prefix="eval_zerodce_dust3r_")
    try:
        for scenario_name in scenarios:
            rows = []
            print(f"\n[scenario] {scenario_name}", flush=True)
            for cat, seq_id in scenes:
                seq_tmp = os.path.join(tmp_root, scenario_name, f"{cat}_{seq_id}")
                os.makedirs(seq_tmp, exist_ok=True)
                img_dir, gt_ply = resolve_sequence_paths(args.co3d_processed, cat, seq_id, args.co3d_raw)
                base_paths = frame_list(img_dir, args.n_frames)
                eval_paths = build_frame_paths(
                    base_paths,
                    scenario_name,
                    zerodce,
                    device,
                    seq_tmp,
                    cat,
                    seq_id,
                    args.brightness,
                    args.gamma,
                    args.noise_seed,
                )
                cd = infer_sequence_cd(
                    dust3r,
                    device,
                    eval_paths,
                    gt_ply,
                    args.image_size,
                    args.align_niter,
                )
                row = {
                    "scenario": scenario_name,
                    "category": cat,
                    "seq_id": seq_id,
                    "chamfer": float(cd),
                }
                rows.append(row)
                all_rows.append(row)
                print(f"  {cat}/{seq_id} CD={cd:.8f}", flush=True)
            mean_by_cat, mean_all = summarize_rows(rows)
            summary[scenario_name] = {"mean_by_category": mean_by_cat, "mean_all": mean_all}

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    result = {
        "split": args.split,
        "co3d_processed": args.co3d_processed,
        "student_ckpt": args.student_ckpt,
        "zerodce_root": args.zerodce_root,
        "dust3r_ckpt": args.dust3r_ckpt,
        "synthetic_lowlight": {"brightness": args.brightness, "gamma": args.gamma, "noise_seed": args.noise_seed},
        "n_frames": args.n_frames,
        "image_size": args.image_size,
        "align_niter": args.align_niter,
        "categories": args.categories,
        "scenes": len(scenes),
        "summary": summary,
        "rows": all_rows,
    }
    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Average Chamfer (lower is better) ===")
    print("| Scenario | Mean Chamfer |")
    print("|---:|---:|")
    for name in scenarios:
        print(f"| {name} | {summary[name]['mean_all']:.8f} |")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
