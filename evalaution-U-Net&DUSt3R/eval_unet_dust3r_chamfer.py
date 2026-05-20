#!/usr/bin/env python3
"""Evaluate DUSt3R vs Student+DUSt3R with clean/motion/defocus inputs via Chamfer distance."""

import argparse
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
for p in (os.path.join(PROJECT_DIR, "dust3r"), PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401
from restoration_kd_ysu.train_kd_restormer_frontend import StudentRestorationFrontEnd

EVAL_CATEGORIES = ["bottle", "cup", "donut", "teddybear", "couch", "toytrain"]


def motion_kernel_25() -> np.ndarray:
    k = np.zeros((25, 25), dtype=np.float32)
    k[12, :] = 1.0
    k /= k.sum()
    return k


def defocus_kernel_r7() -> np.ndarray:
    r = 7
    s = 2 * r + 1
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x * x + y * y) <= (r * r)
    k = np.zeros((s, s), dtype=np.float32)
    k[mask] = 1.0
    k /= k.sum()
    return k


MOTION_K = motion_kernel_25()
DEFOCUS_K = defocus_kernel_r7()


def apply_motion_blur(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img_bgr, -1, MOTION_K)


def apply_defocus_blur(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img_bgr, -1, DEFOCUS_K)


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
        os.path.join(images_dir, x) for x in os.listdir(images_dir) if x.lower().endswith((".jpg", ".jpeg", ".png"))
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


def run_student_on_bgr(student, device, bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    h, w = t.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        out = student(t).clamp(0.0, 1.0)
    out = out[:, :, :h, :w]
    out_np = out[0].permute(1, 2, 0).cpu().numpy()
    out_bgr = cv2.cvtColor((out_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr


def build_eval_images(img_paths, mode: str, use_student: bool, student, device, tmp_root: str):
    out_paths = []
    for i, p in enumerate(img_paths):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")

        if mode == "clean":
            proc = bgr
        elif mode == "motion":
            proc = apply_motion_blur(bgr)
        elif mode == "defocus":
            proc = apply_defocus_blur(bgr)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if use_student:
            proc = run_student_on_bgr(student, device, proc)

        dst = os.path.join(tmp_root, f"f_{i:03d}.jpg")
        cv2.imwrite(dst, proc)
        out_paths.append(dst)
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


def summarize_rows(rows):
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["chamfer"])
    mean_by_cat = {k: float(np.mean(v)) for k, v in by_cat.items()}
    mean_all = float(np.mean([r["chamfer"] for r in rows])) if rows else float("nan")
    return mean_by_cat, mean_all


def main():
    ap = argparse.ArgumentParser("Evaluate DUSt3R and Student+DUSt3R Chamfer table")
    ap.add_argument("--co3d_processed", required=True)
    ap.add_argument("--co3d_raw", default=None)
    ap.add_argument("--split", default="test_10cat8")
    ap.add_argument("--categories", nargs="+", default=EVAL_CATEGORIES)
    ap.add_argument("--dust3r_ckpt", required=True)
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_data = load_split_json(args.co3d_processed, args.split)
    scenes = list(iter_scenes(split_data, set(args.categories)))
    if not scenes:
        raise RuntimeError(f"No scenes for categories {args.categories} in split {args.split}")

    from dust3r.model import load_model

    print(f"[startup] device={device} scenes={len(scenes)} n_frames={args.n_frames}", flush=True)
    print(f"[startup] categories={args.categories}", flush=True)

    dust3r = load_model(args.dust3r_ckpt, device="cpu", verbose=False).to(device).eval()

    student = StudentRestorationFrontEnd().to(device).eval()
    ck = torch.load(args.student_ckpt, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    student.load_state_dict(state, strict=True)
    for p in student.parameters():
        p.requires_grad = False

    scenarios = [
        ("dust3r_clean", "clean", False),
        ("dust3r_motion_blur", "motion", False),
        ("dust3r_defocus_blur", "defocus", False),
        ("unet_dust3r_clean", "clean", True),
        ("unet_dust3r_motion_blur", "motion", True),
        ("unet_dust3r_defocus_blur", "defocus", True),
    ]

    all_rows = []
    summary = {}

    tmp_root = tempfile.mkdtemp(prefix="eval_unet_dust3r_")
    try:
        for scenario_name, blur_mode, use_student in scenarios:
            rows = []
            print(f"\n[scenario] {scenario_name} blur={blur_mode} use_student={use_student}", flush=True)
            for cat, seq_id in scenes:
                seq_tmp = os.path.join(tmp_root, scenario_name, f"{cat}_{seq_id}")
                os.makedirs(seq_tmp, exist_ok=True)
                img_dir, gt_ply = resolve_sequence_paths(args.co3d_processed, cat, seq_id, args.co3d_raw)
                base_paths = frame_list(img_dir, args.n_frames)
                eval_paths = build_eval_images(base_paths, blur_mode, use_student, student, device, seq_tmp)
                cd = infer_sequence_cd(dust3r, device, eval_paths, gt_ply, args.image_size)
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
        "n_frames": args.n_frames,
        "categories": args.categories,
        "scenes": len(scenes),
        "summary": summary,
        "rows": all_rows,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Average Chamfer Table (lower is better) ===")
    print("| Scenario | Average Chamfer |")
    print("|---|---:|")
    order = [s[0] for s in scenarios]
    for name in order:
        print(f"| {name} | {summary[name]['mean_all']:.8f} |")

    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
