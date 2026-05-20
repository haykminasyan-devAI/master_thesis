#!/usr/bin/env python3
import argparse
import os
import sys

import cv2
import numpy as np
import torch


def make_disk_kernel(radius: int) -> np.ndarray:
    k = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    mask = (xx * xx + yy * yy) <= (radius * radius)
    ker = mask.astype(np.float32)
    s = ker.sum()
    if s <= 0:
        return np.ones((1, 1), np.float32)
    return ker / s


def apply_defocus_blur(img_bgr: np.ndarray, radius: int) -> np.ndarray:
    ker = make_disk_kernel(radius)
    out = cv2.filter2D(img_bgr, -1, ker, borderType=cv2.BORDER_REFLECT)
    return out


def load_uformer(uformer_repo: str, weights_path: str):
    uformer_repo = os.path.abspath(uformer_repo)
    if uformer_repo not in sys.path:
        sys.path.insert(0, uformer_repo)
    model_py = os.path.join(uformer_repo, "model.py")
    if not os.path.isfile(model_py):
        raise FileNotFoundError(f"Uformer model.py not found: {model_py}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("uformer_repo_model", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Uformer = module.Uformer

    net = Uformer(
        img_size=128,
        embed_dim=32,
        win_size=8,
        token_projection='linear',
        token_mlp='leff',
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        modulator=True,
        dd_in=3,
    )

    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in state.items():
        clean[k[7:]] = v if k.startswith("module.") else v
    missing, unexpected = net.load_state_dict(clean, strict=False)
    print(f"Loaded Uformer weights: {weights_path}")
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")
    return net


@torch.no_grad()
def uformer_deblur(img_bgr: np.ndarray, net, device: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    t128 = torch.nn.functional.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)
    out = net(t128).clamp(0, 1)
    out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def save_compare(orig_bgr, blur_bgr, deblur_bgr, out_path):
    labels = ["Original", "Defocus blur", "Uformer deblur"]
    tiles = [orig_bgr, blur_bgr, deblur_bgr]
    h, w = orig_bgr.shape[:2]
    banner = np.zeros((36, w * 3, 3), dtype=np.uint8)
    for i, txt in enumerate(labels):
        cv2.putText(
            banner, txt, (i * w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (220, 220, 220), 2, cv2.LINE_AA
        )
    row = np.concatenate(tiles, axis=1)
    comp = np.concatenate([banner, row], axis=0)
    cv2.imwrite(out_path, comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--out_dir", default="outputs/defocusblur_res/teddybear")
    ap.add_argument("--radius", type=int, default=6, help="defocus disk radius")
    ap.add_argument("--uformer_repo", default="Uformer")
    ap.add_argument("--uformer_weights", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image_path}")

    blur = apply_defocus_blur(img, args.radius)
    cv2.imwrite(os.path.join(args.out_dir, "frame_original.jpg"), img)
    cv2.imwrite(os.path.join(args.out_dir, f"frame_defocus_r{args.radius}.jpg"), blur)

    net = load_uformer(args.uformer_repo, args.uformer_weights).to(args.device).eval()
    deblur = uformer_deblur(blur, net, args.device)
    cv2.imwrite(os.path.join(args.out_dir, "frame_defocus_uformer_deblur.jpg"), deblur)

    cmp_path = os.path.join(args.out_dir, "defocus_uformer_compare.jpg")
    save_compare(img, blur, deblur, cmp_path)
    print(f"Saved:\n- {args.out_dir}/frame_original.jpg")
    print(f"- {args.out_dir}/frame_defocus_r{args.radius}.jpg")
    print(f"- {args.out_dir}/frame_defocus_uformer_deblur.jpg")
    print(f"- {cmp_path}")


if __name__ == "__main__":
    main()
