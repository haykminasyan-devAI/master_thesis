"""Directional (linear) motion blur only — no other augmentations."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DirectionalMotionBlur:
    kernel_min: int = 3
    kernel_max: int = 9

    def sample_kernel_size(self, rng: np.random.Generator) -> int:
        k = int(rng.integers(self.kernel_min, self.kernel_max + 1))
        if k % 2 == 0:
            k = k + 1 if k < self.kernel_max else k - 1
        return max(3, k)


def _linear_motion_kernel(kernel_size: int, theta: float) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    half = center
    dx = float(np.cos(theta) * half)
    dy = float(np.sin(theta) * half)
    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, p1, p2, color=1.0, thickness=1)
    s = float(kernel.sum())
    if s <= 0.0:
        kernel[int(center), int(center)] = 1.0
        s = 1.0
    return kernel / s


def apply_directional_blur_to_rgb(
    rgb_uint8: np.ndarray,
    rng: np.random.Generator,
    cfg: DirectionalMotionBlur,
) -> np.ndarray:
    """rgb_uint8: H×W×3 uint8 (RGB or BGR — same convolution per channel)."""
    arr = np.asarray(rgb_uint8, dtype=np.float32)
    k = cfg.sample_kernel_size(rng)
    theta = float(rng.uniform(0.0, np.pi))
    kernel = _linear_motion_kernel(k, theta)
    out = cv2.filter2D(arr, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
