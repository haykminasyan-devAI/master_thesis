# Co3dBlur dataset: same as Co3d but loads blurred images
# from outputs/degraded_frames/ instead of clean images.
# Depth, pose, intrinsics are unchanged from clean CO3D.

import os.path as osp 
import sys

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../../dust3r')))

from dust3r.datasets.co3d import Co3d

class Co3dBlur(Co3d):
    """
    DUSt3R fine-tuning dataset with Gaussian-blurred images.
    Args:
      blur_sigma  : blur level used, e.g. 5, 10, or 20
      blur_root   : root folder containing degraded frames,
                    e.g. "outputs/degraded_frames"
                    expected structure:
                      <blur_root>/<category>/<seq_id>/blur_s<sigma>/frame*.jpg
      ROOT        : preprocessed CO3D root (same as Co3d),
                    used for depth / pose / intrinsics .npz files
      all other args/kwargs passed to Co3d as-is
    """

    def __init__(self, blur_sigma, blur_root, *args, **kwargs):
        self.blur_sigma = blur_sigma
        self.blur_root = osp.abspath(blur_root)
        super().__init__(*args, **kwargs)

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(
            self.blur_root,
            obj,
            instance,
            f'blur_s{self.blur_sigma}',
            f'frame{view_idx:06d}.jpg',
        )






