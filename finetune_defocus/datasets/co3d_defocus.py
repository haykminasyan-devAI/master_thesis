import os.path as osp
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../../dust3r")))
from dust3r.datasets.co3d import Co3d


def _disk_kernel(radius: int) -> np.ndarray:
    r = int(radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    m = (xx * xx + yy * yy) <= (r * r)
    k = m.astype(np.float32)
    return k / max(k.sum(), 1.0)


def _apply_defocus(rgb: np.ndarray, radius: int) -> np.ndarray:
    k = _disk_kernel(radius)
    return cv2.filter2D(rgb, -1, k, borderType=cv2.BORDER_REFLECT)


class Co3dDefocus(Co3d):
    """
    Synthetic defocus blur wrapper over processed CO3D.
    Uses clean CO3D RGB and applies isotropic disk blur on the fly.

    - defocus_radius: blur radius for val/test (and for train if train min/max unset).
    - defocus_train_radius_min / max: if both set and min < max, train split samples an
      integer radius in [min, max] per stereo pair (same radius for both views in the pair).
    """

    def __init__(
        self,
        defocus_radius: int,
        defocus_train_radius_min=None,
        defocus_train_radius_max=None,
        *args,
        **kwargs,
    ):
        self.defocus_radius_eval = int(defocus_radius)
        if defocus_train_radius_min is not None and defocus_train_radius_max is not None:
            lo, hi = int(defocus_train_radius_min), int(defocus_train_radius_max)
            if lo > hi:
                lo, hi = hi, lo
            self.defocus_train_radius_min = lo
            self.defocus_train_radius_max = hi
        else:
            self.defocus_train_radius_min = self.defocus_train_radius_max = self.defocus_radius_eval
        self.defocus_radius = self.defocus_radius_eval
        super().__init__(*args, **kwargs)

    def _sample_blur_radius(self, rng):
        sp = getattr(self, "split", "") or ""
        if (
            str(sp).startswith("train")
            and self.defocus_train_radius_max > self.defocus_train_radius_min
        ):
            return int(rng.integers(self.defocus_train_radius_min, self.defocus_train_radius_max + 1))
        return self.defocus_radius_eval

    def _get_clean_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.jpg")

    def _get_views(self, idx, resolution, rng):
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]

        last = len(image_pool) - 1
        if resolution not in self.invalidate[obj, instance]:
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        mask_bg = (self.mask_bg is True) or (self.mask_bg == "rand" and rng.choice(2))
        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        disk_r = self._sample_blur_radius(rng)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            if self.invalidate[obj, instance][resolution][im_idx]:
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]
            impath_clean = self._get_clean_impath(obj, instance, view_idx)
            depthpath = self._get_depthpath(obj, instance, view_idx)
            metadata_path = self._get_metadatapath(obj, instance, view_idx)
            input_metadata = np.load(metadata_path)
            camera_pose = input_metadata["camera_pose"].astype(np.float32)
            intrinsics0 = input_metadata["camera_intrinsics"].astype(np.float32)

            rgb_clean0 = cv2.cvtColor(cv2.imread(impath_clean, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            rgb_defocus0 = _apply_defocus(rgb_clean0, disk_r)
            depthmap0 = self._read_depthmap(depthpath, input_metadata)

            if mask_bg:
                maskpath = self._get_maskpath(obj, instance, view_idx)
                maskmap = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
                depthmap0 *= maskmap

            rgb_defocus, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_defocus0, depthmap0, intrinsics0, resolution, rng=rng, info=impath_clean
            )

            if (depthmap > 0.0).sum() == 0:
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            views.append(
                dict(
                    img=rgb_defocus,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset=self.dataset_label,
                    label=osp.join(obj, instance),
                    instance=osp.split(impath_clean)[1],
                )
            )

        return views
