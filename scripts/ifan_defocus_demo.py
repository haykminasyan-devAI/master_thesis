#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np


def make_disk_kernel(radius: int) -> np.ndarray:
    k = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    mask = (xx * xx + yy * yy) <= (radius * radius)
    ker = mask.astype(np.float32)
    s = ker.sum()
    return ker / max(s, 1.0)


def apply_defocus_blur(img_bgr: np.ndarray, radius: int) -> np.ndarray:
    ker = make_disk_kernel(radius)
    return cv2.filter2D(img_bgr, -1, ker, borderType=cv2.BORDER_REFLECT)


def save_compare(orig_bgr: np.ndarray, blur_bgr: np.ndarray, deblur_bgr: np.ndarray, out_path: str) -> None:
    labels = ["Clean", "Defocus blur", "IFAN deblur"]
    h, w = orig_bgr.shape[:2]
    banner = np.zeros((36, w * 3, 3), dtype=np.uint8)
    for i, txt in enumerate(labels):
        cv2.putText(
            banner, txt, (i * w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (220, 220, 220), 2, cv2.LINE_AA
        )
    row = np.concatenate([orig_bgr, blur_bgr, deblur_bgr], axis=1)
    comp = np.concatenate([banner, row], axis=0)
    cv2.imwrite(out_path, comp)


def newest_output_image(output_root: str) -> str:
    patterns = [
        os.path.join(output_root, "quanti_quali", "*", "random", "*", "output", "png", "*.png"),
        os.path.join(output_root, "quanti_quali", "*", "random", "*", "output", "jpg", "*.jpg"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return ""
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--ifan_dir", default="/home/asds/project_Hayk_Minasyan/IFAN")
    ap.add_argument("--ifan_ckpt", required=True, help="Path to IFAN.pytorch")
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--out_dir", default="/home/asds/project_Hayk_Minasyan/outputs/defocusblur_res/teddybear_ifan")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cpu", action="store_true", help="Run IFAN on CPU")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_root = os.path.join(args.out_dir, "_ifan_input")
    os.makedirs(os.path.join(tmp_root, "random"), exist_ok=True)

    clean = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if clean is None:
        raise FileNotFoundError(f"Cannot read image: {args.image_path}")
    blur = apply_defocus_blur(clean, args.radius)

    clean_path = os.path.join(args.out_dir, "frame_clean.jpg")
    blur_path = os.path.join(args.out_dir, f"frame_defocus_r{args.radius}.jpg")
    cv2.imwrite(clean_path, clean)
    cv2.imwrite(blur_path, blur)

    # IFAN random mode reads all images from <data_offset>/random
    ifan_input_path = os.path.join(tmp_root, "random", "frame_blur.png")
    cv2.imwrite(ifan_input_path, blur)

    ifan_output_root = os.path.join(args.out_dir, "_ifan_output")
    os.makedirs(ifan_output_root, exist_ok=True)

    cmd = [
        args.python, "run.py",
        "--mode", "IFAN",
        "--network", "IFAN",
        "--config", "config_IFAN",
        "--data", "random",
        "--ckpt_abs_name", args.ifan_ckpt,
        "--data_offset", tmp_root,
        "--output_offset", ifan_output_root,
    ]
    if args.cpu:
        cmd.append("--cpu")

    print("Running IFAN:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=args.ifan_dir, check=True)

    pred_path = newest_output_image(ifan_output_root)
    if not pred_path:
        raise RuntimeError(f"IFAN output image not found under: {ifan_output_root}")
    deblur = cv2.imread(pred_path, cv2.IMREAD_COLOR)
    if deblur is None:
        raise RuntimeError(f"Failed to read IFAN output: {pred_path}")

    if deblur.shape[:2] != clean.shape[:2]:
        deblur = cv2.resize(deblur, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_CUBIC)

    deblur_path = os.path.join(args.out_dir, "frame_defocus_ifan_deblur.jpg")
    cv2.imwrite(deblur_path, deblur)

    compare_path = os.path.join(args.out_dir, "defocus_ifan_compare.jpg")
    save_compare(clean, blur, deblur, compare_path)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] Saved:")
    print(f"- {clean_path}")
    print(f"- {blur_path}")
    print(f"- {deblur_path}")
    print(f"- {compare_path}")
    print(f"- IFAN raw output from: {pred_path}")


if __name__ == "__main__":
    main()
