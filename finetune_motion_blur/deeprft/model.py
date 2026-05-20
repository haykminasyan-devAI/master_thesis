"""
DUSt3R + DeepRFT wrapper for motion-blur finetuning.

Pipeline: motion_blurred_img --> [DeepRFT] --> restored_img --> DUSt3R
"""

import os.path as osp
import sys
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../../dust3r")))
from dust3r.model import load_model


def _set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


def _strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _extract_state(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "params"):
            inner = raw.get(key)
            if isinstance(inner, dict):
                return _strip_module_prefix(inner)
    if isinstance(raw, dict):
        return _strip_module_prefix(raw)
    raise ValueError("Unsupported checkpoint format for DeepRFT weights")


def _load_deeprft(deeprft_repo, weights_path=None, num_res=8):
    if deeprft_repo not in sys.path:
        sys.path.insert(0, deeprft_repo)
    from DeepRFT_MIMO import DeepRFT

    # inference=False to keep train-time branches and gradients.
    net = DeepRFT(num_res=num_res, inference=False)

    if weights_path and osp.isfile(weights_path):
        try:
            raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(weights_path, map_location="cpu")
        state = _extract_state(raw)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"Loaded DeepRFT weights from {weights_path}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    return net


class Dust3rWithDeepRFT(nn.Module):
    def __init__(
        self,
        dust3r_ckpt,
        deeprft_repo,
        deeprft_weights=None,
        device="cuda",
        freeze="deeprft_only",
        use_grad_checkpoint=True,
        deeprft_num_res=8,
        frontend_checkpoint=False,
    ):
        super().__init__()
        self.dust3r = load_model(dust3r_ckpt, device=device)
        self.deeprft = _load_deeprft(
            deeprft_repo=deeprft_repo,
            weights_path=deeprft_weights,
            num_res=deeprft_num_res,
        )
        self.use_grad_checkpoint = bool(use_grad_checkpoint)
        self.frontend_checkpoint = bool(frontend_checkpoint)
        self.set_freeze(freeze)

    def set_freeze(self, mode):
        valid = ("deeprft_only", "deeprft_and_decoder", "all")
        if mode not in valid:
            raise ValueError(f"freeze must be one of {valid}, got {mode!r}")
        self.freeze_mode = mode
        if mode == "deeprft_only":
            _set_requires_grad(self.dust3r, False)
            _set_requires_grad(self.deeprft, True)
        elif mode == "deeprft_and_decoder":
            _set_requires_grad(self.dust3r, False)
            # unfreeze DUSt3R decoder and heads
            for name, p in self.dust3r.named_parameters():
                if any(x in name for x in ("decoder_embed", "dec_blocks", "dec_norm", "head")):
                    p.requires_grad = True
            _set_requires_grad(self.deeprft, True)
        else:
            _set_requires_grad(self.dust3r, True)
            _set_requires_grad(self.deeprft, True)

    def _apply_frontend(self, view):
        v = dict(view)
        img = v["img"]  # DUSt3R image range [-1, 1]
        x = (img + 1.0) * 0.5  # -> [0,1]

        if self.frontend_checkpoint:
            def _fn(inp):
                out = self.deeprft(inp)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                return out
            out = checkpoint(_fn, x, use_reentrant=False)
        else:
            out = self.deeprft(x)
            if isinstance(out, (list, tuple)):
                out = out[0]

        out = torch.clamp(out, 0.0, 1.0)
        v["img"] = out * 2.0 - 1.0  # back to [-1,1]
        return v

    def forward(self, view1, view2):
        view1 = self._apply_frontend(view1)
        view2 = self._apply_frontend(view2)
        return self.dust3r(view1, view2)


def build_model(
    dust3r_ckpt,
    deeprft_repo,
    deeprft_weights=None,
    device="cuda",
    freeze="deeprft_only",
    use_grad_checkpoint=True,
    deeprft_num_res=8,
    frontend_checkpoint=False,
):
    return Dust3rWithDeepRFT(
        dust3r_ckpt=dust3r_ckpt,
        deeprft_repo=deeprft_repo,
        deeprft_weights=deeprft_weights,
        device=device,
        freeze=freeze,
        use_grad_checkpoint=use_grad_checkpoint,
        deeprft_num_res=deeprft_num_res,
        frontend_checkpoint=frontend_checkpoint,
    )
