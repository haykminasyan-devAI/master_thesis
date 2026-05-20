import os.path as osp
import sys

import cv2

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../../dust3r")))

from dust3r.datasets.co3d import Co3d


class Co3dMotion(Co3d):
    """
    DUSt3R fine-tuning dataset for motion-blurred frames.

    Expected layout:
      <motion_root>/<category>/<seq_id>/<motion_tag>/frameXXXXXX.jpg
    """

    def __init__(self, motion_root, motion_tag, *args, **kwargs):
        self.motion_root = osp.abspath(motion_root)
        self.motion_tag = str(motion_tag)
        super().__init__(*args, **kwargs)

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(
            self.motion_root,
            obj,
            instance,
            self.motion_tag,
            f"frame{view_idx:06d}.jpg",
        )

    def _align_rgb_depth_if_needed(self, rgb_image, depthmap):
        """Motion-blur frames may be saved at a different resolution than CO3D depth."""
        if rgb_image is None or depthmap is None:
            return rgb_image, depthmap
        dh, dw = depthmap.shape[0], depthmap.shape[1]
        ih, iw = rgb_image.shape[0], rgb_image.shape[1]
        if (dh, dw) == (ih, iw):
            return rgb_image, depthmap
        interp = cv2.INTER_AREA if (iw * ih) > (dw * dh) else cv2.INTER_LINEAR
        rgb_resized = cv2.resize(rgb_image, (dw, dh), interpolation=interp)
        return rgb_resized, depthmap
