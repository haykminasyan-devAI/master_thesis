"""CO3D frames + synthetic low-light + geometric augmentation + aligned crop."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic_lowlight import create_synthetic_low_light

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_split_json(root: Path, split: str) -> Path:
    candidates = [root / f"selected_seqs_{split}.json"]
    if "_" in split:
        candidates.append(root / f"selected_seqs_{split.split('_', 1)[0]}.json")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Missing split json for:\n - " + "\n - ".join(map(str, candidates)))


class Co3dLowLightKD(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        categories: list[str] | None,
        crop_size: int = 256,
        brightness_range: tuple[float, float] = (0.08, 0.18),
        gamma_range: tuple[float, float] = (2.0, 2.4),
        seed: int = 0,
        is_train: bool = True,
    ):
        self.root = Path(root)
        self.crop_size = crop_size
        self.brightness_range = brightness_range
        self.gamma_range = gamma_range
        self.is_train = is_train
        self._rng_base = np.random.default_rng(seed)

        split_json = resolve_split_json(self.root, split)
        data = json.loads(split_json.read_text())
        allowed = None
        if categories is not None:
            allowed = {str(c).strip().lower().replace(" ", "").replace("_", "") for c in categories}

        self.paths: list[str] = []
        for obj, seqs in data.items():
            if not isinstance(seqs, dict):
                continue
            ok = allowed is None or str(obj).strip().lower().replace(" ", "").replace("_", "") in allowed
            if not ok:
                continue
            for sid in seqs.keys():
                img_dir = self.root / obj / sid / "images"
                if not img_dir.is_dir():
                    continue
                for p in sorted(img_dir.iterdir()):
                    if p.suffix.lower() in IMAGE_EXTS and "frame" in p.name.lower():
                        self.paths.append(str(p))
        if not self.paths:
            raise RuntimeError(f"No frames under split {split_json}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng() if self.is_train else self._rng_base

        p = self.paths[idx]
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")

        h0, w0 = bgr.shape[:2]
        ch = cw = self.crop_size
        if self.is_train:
            brightness = rng.uniform(*self.brightness_range)
            gamma = rng.uniform(*self.gamma_range)
            if h0 < ch or w0 < cw:
                bgr = cv2.resize(bgr, (max(cw, w0), max(ch, h0)), interpolation=cv2.INTER_AREA)
                h0, w0 = bgr.shape[:2]
            y0 = rng.integers(0, h0 - ch + 1) if h0 > ch else 0
            x0 = rng.integers(0, w0 - cw + 1) if w0 > cw else 0
            crop = bgr[y0 : y0 + ch, x0 : x0 + cw].copy()

            if rng.random() < 0.5:
                crop = cv2.flip(crop, 1)
            if rng.random() < 0.5:
                crop = cv2.flip(crop, 0)
            angle = rng.uniform(-15.0, 15.0)
            if abs(angle) > 0.1:
                M = cv2.getRotationMatrix2D((cw / 2, ch / 2), float(angle), 1.0)
                crop = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            brightness = 0.12
            gamma = 2.2
            # center crop val
            if h0 < ch or w0 < cw:
                crop = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA)
            else:
                y0 = (h0 - ch) // 2
                x0 = (w0 - cw) // 2
                crop = bgr[y0 : y0 + ch, x0 : x0 + cw].copy()

        low_u8 = create_synthetic_low_light(crop, brightness_factor=brightness, gamma=gamma, rng=rng)
        low_rgb = cv2.cvtColor(low_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(low_rgb).permute(2, 0, 1).contiguous()
        return t


def collate_stack(batch):
    return torch.stack(batch, dim=0)
