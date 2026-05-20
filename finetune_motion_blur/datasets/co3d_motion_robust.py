import os.path as osp
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../../dust3r")))
from dust3r.datasets.co3d import Co3d


@dataclass(frozen=True)
class MotionBlurConfig:
    kernel_min: int = 3
    kernel_max: int = 9

    def sample_kernel_size(self, rng: np.random.Generator) -> int:
        k = int(rng.integers(self.kernel_min, self.kernel_max + 1))
        # Keep odd-sized kernels for symmetric motion blur.
        if k % 2 == 0:
            k = k + 1 if k < self.kernel_max else k - 1
        return max(3, k)


def _linear_motion_kernel(kernel_size: int, theta: float) -> np.ndarray:
    """
    Build a normalized linear motion blur kernel at angle theta.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    half = center
    dx = np.cos(theta) * half
    dy = np.sin(theta) * half
    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, p1, p2, color=1.0, thickness=1)
    s = float(kernel.sum())
    if s <= 0.0:
        kernel[int(center), int(center)] = 1.0
        s = 1.0
    return kernel / s


def _apply_motion_blur_pil(image: Image.Image, rng: np.random.Generator, cfg: MotionBlurConfig) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    k = cfg.sample_kernel_size(rng)
    theta = float(rng.uniform(0.0, np.pi))
    kernel = _linear_motion_kernel(k, theta)
    out = cv2.filter2D(arr, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out)


class Co3dMotionRobust(Co3d):
    """
    Clean CO3D + on-the-fly linear motion blur.

    train_mode:
      - 50% clean-clean
      - 25% blur image A only
      - 25% blur image B only

    eval_mode:
      - clean-clean
      - blur-blur
      - clean-blur   (image A clean, image B blurred)
    """

    def __init__(
        self,
        *args,
        train_mode: bool = True,
        eval_condition: str = "clean-clean",
        kernel_min: int = 3,
        kernel_max: int = 9,
        **kwargs,
    ):
        self.train_mode = bool(train_mode)
        self.eval_condition = str(eval_condition)
        self.blur_cfg = MotionBlurConfig(kernel_min=kernel_min, kernel_max=kernel_max)
        super().__init__(*args, **kwargs)

    def _sample_train_condition(self, rng: np.random.Generator):
        p = float(rng.random())
        if p < 0.50:
            return False, False  # clean-clean
        if p < 0.75:
            return True, False   # blur image A only
        return False, True       # blur image B only

    def _eval_condition_flags(self):
        cond = self.eval_condition
        if cond == "clean-clean":
            return False, False
        if cond == "blur-blur":
            return True, True
        if cond == "clean-blur":
            return False, True
        raise ValueError(f"Unsupported eval_condition={cond!r}")

    def _get_views(self, idx, resolution, rng):
        views = super()._get_views(idx, resolution, rng)
        blur_a, blur_b = self._sample_train_condition(rng) if self.train_mode else self._eval_condition_flags()
        if blur_a:
            views[0]["img"] = _apply_motion_blur_pil(views[0]["img"], rng, self.blur_cfg)
        if blur_b:
            views[1]["img"] = _apply_motion_blur_pil(views[1]["img"], rng, self.blur_cfg)
        return views
