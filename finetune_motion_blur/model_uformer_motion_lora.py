"""Motion-blur only: Uformer + DUSt3R with PEFT LoRA on DUSt3R encoder and Uformer Linears.

DUSt3R decoder (and heads) stay frozen. Does not modify finetune_noise.
"""

import os.path as osp
import sys
from collections import OrderedDict
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../dust3r")))
from dust3r.model import load_model


def _import_uformer_helpers(uformer_repo):
    uformer_repo = osp.abspath(uformer_repo)
    if not osp.isdir(uformer_repo):
        raise FileNotFoundError(f"Uformer repo not found: {uformer_repo}")
    for p in [uformer_repo, osp.join(uformer_repo, "utils")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    uformer_model_path = osp.join(uformer_repo, "model.py")
    if not osp.isfile(uformer_model_path):
        raise FileNotFoundError(f"Uformer model.py not found: {uformer_model_path}")
    spec = importlib.util.spec_from_file_location("uformer_repo_model_motion", uformer_model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Uformer module spec from: {uformer_model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Uformer


def _load_uformer(uformer_repo, weights_path=None):
    Uformer = _import_uformer_helpers(uformer_repo)
    net = Uformer(
        img_size=128,
        embed_dim=32,
        win_size=8,
        token_projection="linear",
        token_mlp="leff",
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        modulator=True,
        dd_in=3,
    )
    if weights_path and osp.isfile(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected Uformer checkpoint format: {type(state)}")
        clean_state = OrderedDict()
        for k, v in state.items():
            if isinstance(k, str) and k.startswith("module."):
                clean_state[k[len("module.") :]] = v
            else:
                clean_state[k] = v
        missing, unexpected = net.load_state_dict(clean_state, strict=False)
        print(f"Loaded Uformer weights from {weights_path}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    return net


def _lora_config_dust3r_encoder(r: int, alpha: int):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["qkv", "proj", "fc1", "fc2"],
        exclude_modules=[
            "dec_blocks",
            "dec_blocks2",
            "decoder_embed",
            "dec_norm",
            "downstream_head",
            "head1",
            "head2",
            "mask_token",
        ],
    )


def _lora_config_uformer_backbone(r: int, alpha: int):
    from peft import LoraConfig

    # Prefer all Linear layers except I/O heads; fall back if this PEFT build rejects it.
    try:
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            exclude_modules=["output_proj", "input_proj"],
        )
    except (TypeError, ValueError):
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=["to_q", "to_kv", "fc1", "fc2", "proj", "qkv"],
            exclude_modules=["output_proj", "input_proj"],
        )


class Dust3rUformerMotionLora(nn.Module):
    """Pretrained DUSt3R + Uformer; LoRA on DUSt3R encoder + Uformer Linears; DUSt3R decoder frozen."""

    def __init__(
        self,
        dust3r_ckpt: str,
        uformer_repo: str,
        uformer_weights=None,
        device: str = "cpu",
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        try:
            from peft import get_peft_model
        except ImportError as e:
            raise ImportError("Motion LoRA training requires: pip install 'peft>=0.11'") from e

        dust3r = load_model(dust3r_ckpt, device=device)
        uformer = _load_uformer(uformer_repo, uformer_weights)

        for p in dust3r.parameters():
            p.requires_grad = False
        for p in uformer.parameters():
            p.requires_grad = False

        dust3r = get_peft_model(dust3r, _lora_config_dust3r_encoder(lora_r, lora_alpha))
        try:
            uformer = get_peft_model(uformer, _lora_config_uformer_backbone(lora_r, lora_alpha))
        except Exception as ex:
            from peft import LoraConfig

            print(f"WARNING: Uformer LoRA all-linear attach failed ({ex}); using explicit target_modules.")
            u_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=["to_q", "to_kv", "fc1", "fc2", "proj", "qkv"],
                exclude_modules=["output_proj", "input_proj"],
            )
            uformer = get_peft_model(uformer, u_cfg)

        self.dust3r = dust3r
        self.uformer = uformer

    def _denoise_view(self, view):
        img = view["img"]
        img01 = (img + 1.0) * 0.5
        h, w = img01.shape[-2:]
        u_in = F.interpolate(img01, size=(128, 128), mode="bilinear", align_corners=False)
        out01 = F.interpolate(
            self.uformer(u_in).clamp(0, 1), size=(h, w), mode="bilinear", align_corners=False
        )
        restored = out01 * 2.0 - 1.0
        view["img_noisy"] = img
        view["img_restored"] = restored
        view["img"] = restored
        return view

    def forward(self, view1, view2):
        view1 = self._denoise_view(view1)
        view2 = self._denoise_view(view2)
        return self.dust3r(view1, view2)


def build_model(
    dust3r_ckpt: str,
    uformer_repo: str,
    uformer_weights=None,
    device: str = "cpu",
    *,
    lora_r: int = 8,
    lora_alpha: int = 16,
):
    """LoRA motion setup only (no dependency on finetune_noise)."""
    return Dust3rUformerMotionLora(
        dust3r_ckpt=dust3r_ckpt,
        uformer_repo=uformer_repo,
        uformer_weights=uformer_weights,
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
