#!/usr/bin/env python3
"""
Chamfer evaluation: vanilla DUSt3R vs DUSt3R + fine-tuned DeblurDiNAT (checkpoint-best-val).

Scenarios:
  - dust3r_clean
  - dust3r_motion_blur
  - dust3r_defocus_blur
  - deblurdinat_best_motion_blur   (motion blur -> DeblurDiNAT -> DUSt3R)
  - deblurdinat_best_defocus_blur  (defocus blur -> DeblurDiNAT -> DUSt3R)
  - deblurdinat_best_clean         (clean -> DeblurDiNAT -> DUSt3R; same scaling as training, no synthetic blur)
"""

from __future__ import annotations

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
import torch.nn.functional as F
from scipy.spatial import KDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEBLUR_SCRIPT_DIR = os.path.join(PROJECT_DIR, "finetuning Motion&Defocus", "deblurdinat")
for p in (os.path.join(PROJECT_DIR, "dust3r"), PROJECT_DIR, DEBLUR_SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dust3r.utils.path_to_croco  # noqa: F401

from model_motion_defocus import build_model  # noqa: E402

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


def _depthwise_blur(x: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    k = kernel_2d.shape[-1]
    pad = k // 2
    w = kernel_2d.view(1, 1, k, k).repeat(3, 1, 1, 1)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_pad, w, stride=1, padding=0, groups=3)


def _motion_kernel_25(device: torch.device) -> torch.Tensor:
    k = torch.zeros((25, 25), dtype=torch.float32, device=device)
    k[12, :] = 1.0
    return k / k.sum()


def _defocus_kernel_r7(device: torch.device) -> torch.Tensor:
    r = 7
    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    mask = (xx * xx + yy * yy) <= (r * r)
    ker = torch.zeros((2 * r + 1, 2 * r + 1), dtype=torch.float32, device=device)
    ker[mask] = 1.0
    return ker / ker.sum()


def bgr_uint8_to_neg1_1(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def neg1_1_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    x = (t[0].detach().permute(1, 2, 0).float().cpu().numpy() + 1.0) * 0.5
    x = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def apply_synthetic_blur_neg1(
    img_neg1: torch.Tensor,
    mode: str,
    motion_k: torch.Tensor,
    defocus_k: torch.Tensor,
) -> torch.Tensor:
    """Match model_motion_defocus._apply_synthetic_blur for a single image (no random)."""
    x = (img_neg1 + 1.0) * 0.5
    if mode == "clean":
        y = x
    elif mode == "motion":
        y = _depthwise_blur(x, motion_k.to(x.device))
    elif mode == "defocus":
        y = _depthwise_blur(x, defocus_k.to(x.device))
    else:
        raise ValueError(mode)
    y = y.clamp(0.0, 1.0)
    return y * 2.0 - 1.0


@torch.no_grad()
def run_deblurdinat_branch(
    deblurdinat: torch.nn.Module,
    img_neg1: torch.Tensor,
    blur_mode: str,
    motion_k: torch.Tensor,
    defocus_k: torch.Tensor,
) -> torch.Tensor:
    """Match model_motion_defocus._apply_deblur (without view dict). blur_mode: clean|motion|defocus."""
    img_blur = apply_synthetic_blur_neg1(img_neg1, blur_mode, motion_k, defocus_k)
    img_dinat = img_blur * 0.5
    # DeblurDiNAT's multi-scale fusion can fail on odd spatial sizes (e.g., 683 vs 684).
    # Pad to a safe multiple, run inference, then crop back.
    h0, w0 = img_dinat.shape[-2:]
    mult = 8
    ph = (mult - (h0 % mult)) % mult
    pw = (mult - (w0 % mult)) % mult
    if ph or pw:
        img_dinat = F.pad(img_dinat, (0, pw, 0, ph), mode="reflect")
    dev = img_neg1.device.type
    with torch.amp.autocast(dev, enabled=False):
        deblurred = deblurdinat(img_dinat.float())
    if ph or pw:
        deblurred = deblurred[:, :, :h0, :w0]
    out = (deblurred * 2.0).to(dtype=img_neg1.dtype)
    return out.clamp(-1.0, 1.0)


def load_ply_xyz(path: str) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(path, process=False)
    return np.asarray(mesh.vertices if hasattr(mesh, "vertices") else mesh.points, dtype=np.float32)


def compute_chamfer(pred: np.ndarray, gt: np.ndarray) -> float:
    tree_gt = KDTree(gt)
    tree_pred = KDTree(pred)
    d_p2g, _ = tree_gt.query(pred)
    d_g2p, _ = tree_pred.query(gt)
    cd_p2g = float(np.mean(d_p2g**2))
    cd_g2p = float(np.mean(d_g2p**2))
    return 0.5 * (cd_p2g + cd_g2p)


def frame_list(images_dir: str, n_frames: int) -> list[str]:
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


def resolve_sequence_paths(co3d_root: str, category: str, seq_id: str, co3d_raw_root: str | None):
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


def load_split_json(root: str, split_name: str) -> dict:
    candidates = [Path(root) / f"selected_seqs_{split_name}.json"]
    if "_" in split_name:
        candidates.append(Path(root) / f"selected_seqs_{split_name.split('_', 1)[0]}.json")
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    looked = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing split json. Looked for:\n - {looked}")


def iter_scenes(data: dict, categories_filter: set[str] | None):
    for cat, seqs in data.items():
        if categories_filter is not None and cat not in categories_filter:
            continue
        if not isinstance(seqs, dict):
            continue
        for sid in sorted(seqs.keys()):
            yield cat, sid


def build_eval_images(
    img_paths: list[str],
    scenario: str,
    device: torch.device,
    finetune_model: torch.nn.Module | None,
    tmp_root: str,
    motion_k: torch.Tensor,
    defocus_k: torch.Tensor,
) -> list[str]:
    out_paths = []
    deblurdinat = finetune_model.deblurdinat if finetune_model is not None else None

    for i, p in enumerate(img_paths):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")

        if scenario == "dust3r_clean":
            proc = bgr
        elif scenario == "dust3r_motion_blur":
            t = bgr_uint8_to_neg1_1(bgr, device)
            t2 = apply_synthetic_blur_neg1(t, "motion", motion_k, defocus_k)
            proc = neg1_1_to_bgr_uint8(t2)
        elif scenario == "dust3r_defocus_blur":
            t = bgr_uint8_to_neg1_1(bgr, device)
            t2 = apply_synthetic_blur_neg1(t, "defocus", motion_k, defocus_k)
            proc = neg1_1_to_bgr_uint8(t2)
        elif scenario == "deblurdinat_best_motion_blur":
            assert deblurdinat is not None
            t = bgr_uint8_to_neg1_1(bgr, device)
            t2 = run_deblurdinat_branch(deblurdinat, t, "motion", motion_k, defocus_k)
            proc = neg1_1_to_bgr_uint8(t2)
        elif scenario == "deblurdinat_best_defocus_blur":
            assert deblurdinat is not None
            t = bgr_uint8_to_neg1_1(bgr, device)
            t2 = run_deblurdinat_branch(deblurdinat, t, "defocus", motion_k, defocus_k)
            proc = neg1_1_to_bgr_uint8(t2)
        elif scenario == "deblurdinat_best_clean":
            assert deblurdinat is not None
            t = bgr_uint8_to_neg1_1(bgr, device)
            t2 = run_deblurdinat_branch(deblurdinat, t, "clean", motion_k, defocus_k)
            proc = neg1_1_to_bgr_uint8(t2)
        else:
            raise ValueError(scenario)

        dst = os.path.join(tmp_root, f"f_{i:03d}.jpg")
        cv2.imwrite(dst, proc)
        out_paths.append(dst)

    return out_paths


def infer_sequence_cd(model, device, img_paths: list[str], gt_ply: str, image_size: int, align_niter: int) -> float:
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


def summarize_rows(rows: list[dict]) -> tuple[dict[str, float], float]:
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["chamfer"])
    mean_by_cat = {k: float(np.mean(v)) for k, v in by_cat.items()}
    mean_all = float(np.mean([r["chamfer"] for r in rows])) if rows else float("nan")
    return mean_by_cat, mean_all


def load_finetuned_model(
    dust3r_ckpt: str,
    deblurdinat_repo: str,
    deblurdinat_weights: str,
    finetuned_ckpt: str,
    device: torch.device,
    motion_prob: float,
) -> torch.nn.Module:
    m = build_model(
        dust3r_ckpt=dust3r_ckpt,
        deblurdinat_repo=deblurdinat_repo,
        deblurdinat_weights=deblurdinat_weights,
        device="cpu",
        freeze="deblurdinat_only",
        use_grad_checkpoint=False,
        deblur_checkpoint=False,
        motion_prob=motion_prob,
    ).to(device)
    try:
        ck = torch.load(finetuned_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(finetuned_ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] strict=False missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def main():
    ap = argparse.ArgumentParser("DeblurDiNAT + DUSt3R Chamfer (6 scenarios)")
    ap.add_argument("--co3d_processed", required=True)
    ap.add_argument("--co3d_raw", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--dust3r_ckpt", required=True)
    ap.add_argument("--deblurdinat_repo", required=True)
    ap.add_argument("--deblurdinat_weights", required=True, help="Initial DeblurDiNATL.pth (for arch + missing keys)")
    ap.add_argument(
        "--finetuned_ckpt",
        required=True,
        help="Fine-tuned checkpoint (e.g. checkpoint-best-val.pth with key 'model')",
    )
    ap.add_argument("--motion_prob", type=float, default=0.5, help="Only used when building the model (must match training).")
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

    motion_k = _motion_kernel_25(device)
    defocus_k = _defocus_kernel_r7(device)

    print(f"[startup] device={device} scenes={len(scenes)} n_frames={args.n_frames}", flush=True)
    print(f"[startup] finetuned_ckpt={args.finetuned_ckpt}", flush=True)

    # Single DUSt3R from fine-tuned checkpoint (encoder frozen during training; weights match base).
    finetuned = load_finetuned_model(
        args.dust3r_ckpt,
        args.deblurdinat_repo,
        args.deblurdinat_weights,
        args.finetuned_ckpt,
        device,
        args.motion_prob,
    )
    dust3r = finetuned.dust3r

    scenarios = [
        ("dust3r_clean", None),
        ("dust3r_motion_blur", None),
        ("dust3r_defocus_blur", None),
        ("deblurdinat_best_motion_blur", finetuned),
        ("deblurdinat_best_defocus_blur", finetuned),
        ("deblurdinat_best_clean", finetuned),
    ]

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}
    tmp_root = tempfile.mkdtemp(prefix="eval_deblurdinat_dust3r_")

    try:
        for scenario_name, ft_model in scenarios:
            rows = []
            print(f"\n[scenario] {scenario_name}", flush=True)
            for cat, seq_id in scenes:
                seq_tmp = os.path.join(tmp_root, scenario_name, f"{cat}_{seq_id}")
                os.makedirs(seq_tmp, exist_ok=True)
                img_dir, gt_ply = resolve_sequence_paths(args.co3d_processed, cat, seq_id, args.co3d_raw)
                base_paths = frame_list(img_dir, args.n_frames)
                eval_paths = build_eval_images(
                    base_paths,
                    scenario_name,
                    device,
                    ft_model,
                    seq_tmp,
                    motion_k,
                    defocus_k,
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

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "split": args.split,
        "co3d_processed": args.co3d_processed,
        "dust3r_ckpt": args.dust3r_ckpt,
        "finetuned_ckpt": args.finetuned_ckpt,
        "deblurdinat_repo": args.deblurdinat_repo,
        "deblurdinat_weights": args.deblurdinat_weights,
        "motion_prob_build": args.motion_prob,
        "n_frames": args.n_frames,
        "image_size": args.image_size,
        "align_niter": args.align_niter,
        "categories": args.categories,
        "scenes": len(scenes),
        "summary": summary,
        "rows": all_rows,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Average Chamfer (lower is better) ===")
    print("| Scenario | Mean Chamfer |")
    print("|---|---:|")
    for name, _ in scenarios:
        print(f"| {name} | {summary[name]['mean_all']:.8f} |")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
