# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Attach DUSt3R encoder/decoder attention key-padding masks from patch grids.
# Mask file: numpy (nh, nw) bool, True = masked patch (cannot be used as attention key).

from __future__ import annotations

import numpy as np
import torch


def attach_attn_key_padding_from_hw_mask(view: dict, mask_hw: np.ndarray, model) -> None:
    """
    view: dict with 'img' (B,3,H,W).
    mask_hw: (nh, nw) bool, True = masked (black-out) patch; grid matches tensor H,W patch tiling.
    model: DUSt3R model (uses model.patch_size).
    Sets view['attn_key_padding_mask'] shape (B, nh*nw) bool True = masked key.
    """
    img = view["img"]
    B, _, H, W = img.shape
    p = int(model.patch_size)
    nh, nw = H // p, W // p
    if mask_hw.shape != (nh, nw):
        raise ValueError(
            f"mask_hw shape {mask_hw.shape} != {(nh, nw)} for img H,W={H},{W} and patch_size={p}"
        )
    flat = torch.as_tensor(mask_hw.reshape(-1), dtype=torch.bool).unsqueeze(0).expand(B, -1).clone()
    view["attn_key_padding_mask"] = flat
