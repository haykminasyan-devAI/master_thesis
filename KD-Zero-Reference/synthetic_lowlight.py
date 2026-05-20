"""On-the-fly synthetic low-light degradation for CO3D frames (BGR uint8)."""

from __future__ import annotations

import cv2
import numpy as np


def create_synthetic_low_light(
    image: np.ndarray,
    brightness_factor: float = 0.1,
    gamma: float = 2.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulates low-light degradation (BGR, 0–255).

    rng: optional numpy Generator for reproducible augmentations.
    """
    if rng is None:
        rng = np.random.default_rng()

    img_float = image.astype(np.float32) / 255.0
    dark_img = np.power(img_float * float(brightness_factor), 1.0 / float(gamma))

    vals = len(np.unique(np.clip(dark_img * 255.0, 0, 255).astype(np.uint8)))
    vals = max(2.0, float(2 ** np.ceil(np.log2(max(vals, 2)))))
    poisson_noisy = rng.poisson(np.clip(dark_img * vals, 0.0, None)).astype(np.float32) / vals

    h, w, c = image.shape
    chroma_noise = rng.normal(0.0, 0.02, size=(max(1, h // 16), max(1, w // 16), c)).astype(np.float32)
    chroma_noise = cv2.resize(chroma_noise, (w, h), interpolation=cv2.INTER_LINEAR)

    final_noisy = np.clip(poisson_noisy + chroma_noise, 0.0, 1.0)
    return (final_noisy * 255.0).astype(np.uint8)
