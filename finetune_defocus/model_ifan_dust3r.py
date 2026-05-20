import os.path as osp
import sys
from types import SimpleNamespace
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../dust3r")))
from dust3r.model import load_model


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _build_ifan_config(device: str):
    return SimpleNamespace(
        device=device,
        ks=3,
        ch=32,
        res_num=2,
        Fs=3,
        wiF=1.5,
        N=17,
    )


def _load_ifan(ifan_repo: str, ifan_ckpt: str, device: str = "cpu"):
    ifan_repo = osp.abspath(ifan_repo)
    if ifan_repo not in sys.path:
        sys.path.insert(0, ifan_repo)
    # Avoid collisions with unrelated top-level `models` modules.
    for key in list(sys.modules.keys()):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]
    IFANNetwork = importlib.import_module("models.archs.IFAN").Network

    cfg = _build_ifan_config(device)
    net = IFANNetwork(cfg)
    ckpt = torch.load(ifan_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in state.items():
        if k.startswith("module.Network."):
            clean[k[len("module.Network."):]] = v
        elif k.startswith("Network."):
            clean[k[len("Network."):]] = v
        elif not (k.startswith("module.reblurNet.") or k.startswith("reblurNet.")):
            clean[k] = v
    net.load_state_dict(clean, strict=False)
    return net


class Dust3rWithIFAN(nn.Module):
    def __init__(self, dust3r_ckpt, ifan_repo, ifan_ckpt, device="cpu", freeze="ifan_only"):
        super().__init__()
        self.dust3r = load_model(dust3r_ckpt, device=device)
        self.ifan = _load_ifan(ifan_repo, ifan_ckpt, device=device)
        self.set_freeze(freeze)

    def set_freeze(self, mode="ifan_only"):
        if mode == "ifan_only":
            _set_requires_grad(self.dust3r, False)
            _set_requires_grad(self.ifan, True)
        elif mode == "all":
            _set_requires_grad(self.dust3r, True)
            _set_requires_grad(self.ifan, True)
        else:
            raise ValueError(f"Unsupported freeze mode: {mode}")

    def _deblur_view(self, view):
        img = view["img"]  # [-1,1]
        img01 = (img + 1.0) * 0.5
        out = self.ifan(img01, is_train=False)["result"].clamp(0, 1)
        view["img_defocus"] = img
        view["img_restored"] = out * 2.0 - 1.0
        view["img"] = view["img_restored"]
        return view

    def forward(self, view1, view2):
        view1 = self._deblur_view(view1)
        view2 = self._deblur_view(view2)
        return self.dust3r(view1, view2)


def build_model(dust3r_ckpt, ifan_repo, ifan_ckpt, device="cpu", freeze="ifan_only"):
    return Dust3rWithIFAN(
        dust3r_ckpt=dust3r_ckpt,
        ifan_repo=ifan_repo,
        ifan_ckpt=ifan_ckpt,
        device=device,
        freeze=freeze,
    )
