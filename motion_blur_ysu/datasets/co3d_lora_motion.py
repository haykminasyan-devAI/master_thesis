"""CO3D processed subset: same-sequence pairs, temporal distance buckets, optional motion blur."""
import os.path as osp
import json

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

from motion_blur_ysu.augmentation.motion_blur import DirectionalMotionBlur, apply_directional_blur_to_rgb


def _sample_pair_indices(pool_len: int, rng: np.random.Generator):
    """40% near / 40% medium / 20% far by |i-j| relative span; same sequence only."""
    last = pool_len - 1
    if last < 1:
        return 0, min(1, last)

    n_span = float(last)
    r = float(rng.random())
    if r < 0.40:
        d_lo, d_hi = 1, max(1, int(0.20 * n_span))
    elif r < 0.80:
        d_lo = max(1, int(0.20 * n_span) + 1)
        d_hi = max(d_lo, int(0.60 * n_span))
    else:
        d_lo = max(1, int(0.60 * n_span) + 1)
        d_hi = int(n_span)

    d_hi = min(d_hi, last)
    d_lo = min(d_lo, d_hi)

    for _ in range(128):
        d = int(rng.integers(d_lo, d_hi + 1))
        i = int(rng.integers(0, pool_len))
        sgn = int(rng.choice([-1, 1]))
        j = i + sgn * d
        if 0 <= j <= last and i != j:
            return i, j

    i = int(rng.integers(0, pool_len))
    j = int(rng.integers(0, pool_len))
    while j == i:
        j = int(rng.integers(0, pool_len))
    return i, j


class Co3dLoraMotion(BaseStereoViewDataset):
    """
    split: e.g. 'train_10cat8seq' -> selected_seqs_train_10cat8seq.json
    allowed_categories: restrict to this set of object names (None = all in JSON)
    train_mode: if True, random pair + 50/25/25 blur policy; if False, use eval_condition
    eval_condition: 'clean-clean' | 'blur-blur' | 'clean-blur'
    """

    def __init__(
        self,
        ROOT,
        split,
        resolution,
        allowed_categories=None,
        mask_bg=True,
        aug_crop=16,
        train_mode=True,
        eval_condition="clean-clean",
        seed=None,
        blur_cfg=None,
        length_factor: int = 256,
    ):
        self.ROOT = ROOT
        self.dataset_label = "Co3d_v2"
        self.mask_bg = mask_bg
        self.train_mode = bool(train_mode)
        self.eval_condition = str(eval_condition)
        self.blur_cfg = blur_cfg or DirectionalMotionBlur(3, 9)
        self.length_factor = int(length_factor)

        candidates = [osp.join(self.ROOT, f"selected_seqs_{split}.json")]
        # Accept canonical names too, e.g. split=train_10cat8seq -> selected_seqs_train.json
        if "_" in split:
            split_head = split.split("_", 1)[0]
            candidates.append(osp.join(self.ROOT, f"selected_seqs_{split_head}.json"))
        path = next((p for p in candidates if osp.isfile(p)), None)
        if path is None:
            looked_for = "\n - ".join(candidates)
            raise FileNotFoundError(
                f"Missing split JSON under {self.ROOT}. Looked for:\n - {looked_for}"
            )
        with open(path, "r") as f:
            scenes = json.load(f)
        scenes = {k: v for k, v in scenes.items() if isinstance(v, dict) and len(v) > 0}
        if allowed_categories is not None:
            keep = set(allowed_categories)
            scenes = {k: v for k, v in scenes.items() if k in keep}
        flat = {(obj, inst): v2 for obj, v in scenes.items() for inst, v2 in v.items()}
        self.scenes = {
            k: v for k, v in flat.items() if isinstance(v, (list, tuple)) and len(v) >= 2
        }
        self.scene_list = list(self.scenes.keys())
        self.invalidate = {scene: {} for scene in self.scene_list}

        super().__init__(split=split, resolution=resolution, aug_crop=aug_crop, seed=seed)

    def __len__(self):
        return max(1, len(self.scene_list)) * max(1, self.length_factor)

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.npz")

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.jpg")

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(
            self.ROOT, obj, instance, "depths", f"frame{view_idx:06n}.jpg.geometric.png"
        )

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, "masks", f"frame{view_idx:06n}.png")

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(
            input_metadata["maximum_depth"]
        )
        return depthmap

    def _align_rgb_depth_if_needed(self, rgb_image, depthmap):
        return rgb_image, depthmap

    def _pair_blur_flags(self, rng: np.random.Generator):
        if self.train_mode:
            p = float(rng.random())
            if p < 0.50:
                return False, False
            if p < 0.75:
                return True, False
            return False, True
        cond = self.eval_condition
        if cond == "clean-clean":
            return False, False
        if cond == "blur-blur":
            return True, True
        if cond == "clean-blur":
            return False, True
        raise ValueError(f"Unknown eval_condition={cond!r}")

    def _load_one_view(
        self,
        obj,
        instance,
        im_idx,
        image_pool,
        resolution,
        rng,
        mask_bg,
        blur: bool,
    ):
        if self.invalidate[obj, instance].get(resolution) is None:
            self.invalidate[obj, instance][resolution] = [False] * len(image_pool)

        while True:
            if self.invalidate[obj, instance][resolution][im_idx]:
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative]:
                        im_idx = tentative
                        break

            view_idx = image_pool[im_idx]
            impath = self._get_impath(obj, instance, view_idx)
            depthpath = self._get_depthpath(obj, instance, view_idx)
            metadata_path = self._get_metadatapath(obj, instance, view_idx)
            input_metadata = np.load(metadata_path)
            camera_pose = input_metadata["camera_pose"].astype(np.float32)
            intrinsics = input_metadata["camera_intrinsics"].astype(np.float32)

            rgb_image = imread_cv2(impath)
            depthmap = self._read_depthmap(depthpath, input_metadata)
            rgb_image, depthmap = self._align_rgb_depth_if_needed(rgb_image, depthmap)

            if blur:
                rgb_image = apply_directional_blur_to_rgb(rgb_image, rng, self.blur_cfg)

            if mask_bg:
                maskpath = self._get_maskpath(obj, instance, view_idx)
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
            )

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                self.invalidate[obj, instance][resolution][im_idx] = True
                im_idx = (im_idx + 1) % len(image_pool)
                continue

            return dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
            )

    def _get_views(self, idx, resolution, rng):
        sid = idx % len(self.scene_list)
        obj, instance = self.scene_list[sid]
        image_pool = self.scenes[obj, instance]
        pool_len = len(image_pool)
        i1, i2 = _sample_pair_indices(pool_len, rng)

        mask_bg = (self.mask_bg is True) or (self.mask_bg == "rand" and rng.choice(2))
        blur_a, blur_b = self._pair_blur_flags(rng)

        v2 = self._load_one_view(
            obj, instance, i2, image_pool, resolution, rng, mask_bg, blur_b
        )
        v1 = self._load_one_view(
            obj, instance, i1, image_pool, resolution, rng, mask_bg, blur_a
        )
        return [v2, v1]
