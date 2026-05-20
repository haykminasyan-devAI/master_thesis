# finetune_noise/datasets/co3d_noise.py
import copy
import os.path as osp
import sys
from collections import deque

import numpy as np

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../../dust3r')))
from dust3r.datasets.co3d import Co3d
from dust3r.utils.image import ImgNorm, imread_cv2


class Co3dNoise(Co3d):
    """
    CO3D dataset wrapper that replaces clean RGB with noisy RGB, while also
    providing the clean RGB tensor for optional reconstruction loss.
    """
    def __init__(self, noise_sigma, noise_root, *args, **kwargs):
        self.noise_sigma = int(noise_sigma)
        self.noise_root = osp.abspath(noise_root)
        super().__init__(*args, **kwargs)

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(
            self.noise_root,
            obj,
            instance,
            f'noise_s{self.noise_sigma}',
            f'frame{view_idx:06d}.jpg',
        )

    def _get_clean_impath(self, obj, instance, view_idx):
        # Original Co3D clean RGB location under processed ROOT.
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

    def _get_views(self, idx, resolution, rng):
        # Same logic as Co3d._get_views, but loads both noisy+clean RGB and
        # ensures identical crop/resize by reusing the RNG state.
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]

        last = len(image_pool) - 1
        if resolution not in self.invalidate[obj, instance]:
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        mask_bg = (self.mask_bg is True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
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

            impath_noisy = self._get_impath(obj, instance, view_idx)
            impath_clean = self._get_clean_impath(obj, instance, view_idx)
            depthpath = self._get_depthpath(obj, instance, view_idx)

            metadata_path = self._get_metadatapath(obj, instance, view_idx)
            input_metadata = np.load(metadata_path)
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics0 = input_metadata['camera_intrinsics'].astype(np.float32)

            rgb_noisy0 = imread_cv2(impath_noisy)
            rgb_clean0 = imread_cv2(impath_clean)
            depthmap0 = self._read_depthmap(depthpath, input_metadata)

            if mask_bg:
                maskpath = self._get_maskpath(obj, instance, view_idx)
                maskmap = imread_cv2(maskpath, 0).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
                depthmap0 *= maskmap

            # Apply identical crop/resize to (noisy, clean) by replaying RNG state.
            rng_state = copy.deepcopy(rng.bit_generator.state)
            rgb_noisy, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_noisy0, depthmap0.copy(), intrinsics0.copy(), resolution, rng=rng, info=impath_noisy
            )
            rng.bit_generator.state = rng_state
            rgb_clean, _depth_unused, _intr_unused = self._crop_resize_if_necessary(
                rgb_clean0, depthmap0.copy(), intrinsics0.copy(), resolution, rng=rng, info=impath_clean
            )

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            # rgb_noisy is PIL.Image after crop/resize; BaseStereoViewDataset will transform it.
            # rgb_clean is also PIL.Image; we pre-transform it to a tensor so it survives checks.
            img_clean = ImgNorm(rgb_clean)  # CHW float in [-1,1]

            views.append(dict(
                img=rgb_noisy,
                img_clean=img_clean,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=osp.join(obj, instance),
                instance=osp.split(impath_noisy)[1],
            ))

        return views