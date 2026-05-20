"""Load Zero-DCE student (enhance_net_nopool) from cloned Zero-DCE repo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch.nn as nn


def load_zerodce_student(zerodce_root: str) -> nn.Module:
    root = Path(zerodce_root).resolve()
    model_py = root / "Zero-DCE_code" / "model.py"
    if not model_py.is_file():
        raise FileNotFoundError(
            f"Expected {model_py}. Clone https://github.com/Li-Chongyi/Zero-DCE "
            f"and point --zerodce_root to the repo root."
        )
    spec = importlib.util.spec_from_file_location("zerodce_model_dynamic", model_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    net = mod.enhance_net_nopool()
    return net
