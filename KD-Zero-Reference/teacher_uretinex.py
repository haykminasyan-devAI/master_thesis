"""Load frozen URetinex-Net teacher from cloned repo (see README)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

_torch_load_orig = torch.load


def _torch_load_compat(*args, **kwargs):
    """PyTorch 2.6+ defaults weights_only=True; URetinex ckpts store argparse.Namespace etc."""
    kwargs.setdefault("weights_only", False)
    try:
        return _torch_load_orig(*args, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return _torch_load_orig(*args, **kwargs)


def load_uretinex_teacher(uretinex_root: str, device: torch.device, ratio: float = 5.0):
    root = Path(uretinex_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"URetinex root not found: {root}")

    test_py = root / "test.py"
    if not test_py.is_file():
        raise FileNotFoundError(f"Expected {test_py} from https://github.com/AndersonYong/URetinex-Net")

    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("uretinex_test_dynamic", test_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    opts = argparse.Namespace(
        img_path=str(root / "demo/input/3.png"),
        output=str(root / "demo/output"),
        ratio=float(ratio),
        Decom_model_low_path=str(root / "ckpt/init_low.pth"),
        unfolding_model_path=str(root / "ckpt/unfolding.pth"),
        adjust_model_path=str(root / "ckpt/L_adjust.pth"),
        gpu_id=0,
    )

    for ck in (
        opts.Decom_model_low_path,
        opts.unfolding_model_path,
        opts.adjust_model_path,
    ):
        if not Path(ck).is_file():
            raise FileNotFoundError(f"Missing URetinex checkpoint (clone repo + ckpt/): {ck}")

    torch.load = _torch_load_compat
    try:
        model = mod.Inference(opts)
    finally:
        torch.load = _torch_load_orig
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def forward_fixed(self, input_low_img: torch.Tensor) -> torch.Tensor:
        input_low_img = input_low_img.to(device)
        with torch.no_grad():
            R, L = self.unfolding(input_low_img)
            ratio_t = torch.ones_like(L) * float(self.opts.ratio)
            High_L = self.adjust_model(l=L, alpha=ratio_t)
            I_enhance = High_L * R
        return torch.clamp(I_enhance, 0.0, 1.0)

    model.forward = types.MethodType(forward_fixed, model)
    return model
