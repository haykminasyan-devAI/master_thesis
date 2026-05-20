#!/usr/bin/env python3
"""Evaluate KD-Encoder DUSt3R checkpoints with Chamfer distance on 6 categories."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from scipy.spatial import KDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
for p in (os.path.join(PROJECT_DIR, "dust3r"), PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.model import load_model as load_dust3r_model

EVAL_CATEGORIES = ["bottle", "cup", "donut", "teddybear", "couch", "toytrain"]


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


def apply_gamma_darkening(rgb_u8: np.ndarray, gamma: float) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    dark = np.power(rgb, float(gamma))
    return (dark * 255.0).clip(0, 255).astype(np.uint8)


def build_eval_images(img_paths, dark_gamma: float | None, tmp_root: str):
    out_paths = []
    for i, p in enumerate(img_paths):
        try:
            rgb = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to read image: {p}") from e
        proc = rgb if dark_gamma is None else apply_gamma_darkening(rgb, dark_gamma)
        dst = os.path.join(tmp_root, f"f_{i:03d}.jpg")
        Image.fromarray(proc, mode="RGB").save(dst, format="JPEG", quality=95)
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


def build_kd_model(base_dust3r_ckpt: str, kd_ckpt: str, lora_r: int, lora_alpha: int, device):
    model = load_dust3r_model(base_dust3r_ckpt, device="cpu", verbose=False)
    cfg = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        target_modules=["qkv"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    ck = torch.load(kd_ckpt, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[load_kd_model] ckpt={kd_ckpt} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    ap = argparse.ArgumentParser("Evaluate KD-Encoder DUSt3R variants with Chamfer")
    ap.add_argument("--co3d_processed", required=True)
    ap.add_argument("--co3d_raw", default=None)
    ap.add_argument("--split", default="test_10cat8")
    ap.add_argument("--categories", nargs="+", default=EVAL_CATEGORIES)
    ap.add_argument("--dust3r_ckpt", required=True, help="Base DUSt3R checkpoint path")
    ap.add_argument("--kd20_ckpt", required=True, help="KD LoRA checkpoint from 20-epoch run")
    ap.add_argument("--kd50_ckpt", required=True, help="KD LoRA checkpoint from 50-epoch run")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--dark_gammas", nargs="+", type=float, default=[1.5, 2.2])
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
    print(f"[startup] dark_gammas={args.dark_gammas}", flush=True)

    base_model = load_dust3r_model(args.dust3r_ckpt, device="cpu", verbose=False).to(device).eval()
    kd20_model = build_kd_model(args.dust3r_ckpt, args.kd20_ckpt, args.lora_r, args.lora_alpha, device)
    kd50_model = build_kd_model(args.dust3r_ckpt, args.kd50_ckpt, args.lora_r, args.lora_alpha, device)

    scenarios = [
        ("dust3r_clean", base_model, None),
        ("dust3r_dark", base_model, "dark"),
        ("ourdust3r_kd20_dark", kd20_model, "dark"),
        ("ourdust3r_kd50_dark", kd50_model, "dark"),
        ("ourdust3r_kd50_clean", kd50_model, None),
    ]

    all_rows = []
    summary = {}
    tmp_root = tempfile.mkdtemp(prefix="eval_kd_encoder_dust3r_")
    try:
        for scenario_name, model, dark_mode in scenarios:
            rows = []
            print(f"\n[scenario] {scenario_name}", flush=True)
            for cat, seq_id in scenes:
                img_dir, gt_ply = resolve_sequence_paths(args.co3d_processed, cat, seq_id, args.co3d_raw)
                base_paths = frame_list(img_dir, args.n_frames)

                gamma_cds = []
                gammas = args.dark_gammas if dark_mode == "dark" else [None]
                for gamma in gammas:
                    tag = "clean" if gamma is None else f"g{str(gamma).replace('.', 'p')}"
                    seq_tmp = os.path.join(tmp_root, scenario_name, f"{cat}_{seq_id}_{tag}")
                    os.makedirs(seq_tmp, exist_ok=True)
                    eval_paths = build_eval_images(base_paths, gamma, seq_tmp)
                    cd = infer_sequence_cd(model, device, eval_paths, gt_ply, args.image_size, args.align_niter)
                    gamma_cds.append(float(cd))

                cd_avg = float(np.mean(gamma_cds))
                row = {
                    "scenario": scenario_name,
                    "category": cat,
                    "seq_id": seq_id,
                    "chamfer": cd_avg,
                    "gammas": args.dark_gammas if dark_mode == "dark" else ["clean"],
                    "gamma_cds": gamma_cds,
                }
                rows.append(row)
                all_rows.append(row)
                print(f"  {cat}/{seq_id} CD={cd_avg:.8f}", flush=True)

            mean_by_cat, mean_all = summarize_rows(rows)
            summary[scenario_name] = {"mean_by_category": mean_by_cat, "mean_all": mean_all}

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    result = {
        "split": args.split,
        "n_frames": args.n_frames,
        "categories": args.categories,
        "scenes": len(scenes),
        "dark_gammas": args.dark_gammas,
        "summary": summary,
        "rows": all_rows,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Average Chamfer Table (lower is better) ===")
    print("| Scenario | Average Chamfer |")
    print("|---|---:|")
    for name, _, _ in scenarios:
        print(f"| {name} | {summary[name]['mean_all']:.8f} |")
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()

