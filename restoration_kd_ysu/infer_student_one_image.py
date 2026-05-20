#!/usr/bin/env python3
"""Run StudentRestorationFrontEnd on one image (RGB [0,1], pad to /8 like training)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_kd_restormer_frontend import StudentRestorationFrontEnd


def load_student(ckpt_path: Path, device: torch.device) -> StudentRestorationFrontEnd:
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    net = StudentRestorationFrontEnd().to(device).eval()
    net.load_state_dict(state, strict=True)
    return net


def infer_image(net: StudentRestorationFrontEnd, bgr: np.ndarray, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h0, w0 = rgb.shape[0], rgb.shape[1]
    h = int(math.ceil(h0 / 8.0) * 8)
    w = int(math.ceil(w0 / 8.0) * 8)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    if h != h0 or w != w0:
        x = F.pad(x, (0, w - w0, 0, h - h0), mode="reflect")
    with torch.no_grad():
        y = net(x)
    y = y[:, :, :h0, :w0].clamp(0.0, 1.0)
    out_rgb = (y[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] CUDA unavailable, using CPU")
        dev = "cpu"
    device = torch.device(dev)

    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {args.input}")

    net = load_student(args.ckpt, device)
    out_bgr = infer_image(net, bgr, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), out_bgr):
        raise SystemExit(f"Failed to write: {args.output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
