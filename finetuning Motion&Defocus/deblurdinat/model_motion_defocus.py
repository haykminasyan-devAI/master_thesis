"""
DUSt3R + DeblurDiNAT wrapper with synthetic motion/defocus blur corruption.

Pipeline:
clean_img -> synthetic blur (motion or defocus) -> DeblurDiNAT -> DUSt3R -> loss
"""

import inspect
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../../dust3r")))
from dust3r.model import load_model


def _set_requires_grad(module, requires_grad):
    if not hasattr(module, "parameters"):
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


def _deblurdinat_state_dict_from_checkpoint(ckpt):
    state = ckpt
    if isinstance(state, dict):
        for k in ("state_dict", "model", "params"):
            inner = state.get(k)
            if isinstance(inner, dict) and inner and all(isinstance(x, str) for x in inner):
                state = inner
                break
    if not isinstance(state, dict):
        raise ValueError("DeblurDiNAT weights must be a state_dict or dict containing one")
    out = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def _load_deblurdinat(deblurdinat_repo, weights_path=None):
    models_dir = osp.join(deblurdinat_repo, "models")
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    if deblurdinat_repo not in sys.path:
        sys.path.insert(0, deblurdinat_repo)

    from DeblurDiNATL import NADeblurPlus

    net = NADeblurPlus()
    if weights_path and osp.isfile(weights_path):
        try:
            raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(weights_path, map_location="cpu")
        state = _deblurdinat_state_dict_from_checkpoint(raw)
        # Pretrained DeblurDiNATL checkpoints often predate NATTEN's per-layer `rpb`
        # buffers; the live module may define them. strict=False loads all matching
        # weights and leaves missing tensors at their default init.
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"Loaded DeblurDiNAT from {weights_path} (strict=False): "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
            )
        else:
            print(f"Loaded DeblurDiNAT weights from {weights_path}")
    return net


def _motion_kernel_25(device):
    k = torch.zeros((25, 25), dtype=torch.float32, device=device)
    k[12, :] = 1.0
    k = k / k.sum()
    return k


def _defocus_kernel_r7(device):
    r = 7
    s = 2 * r + 1
    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    mask = (xx * xx + yy * yy) <= (r * r)
    k = torch.zeros((s, s), dtype=torch.float32, device=device)
    k[mask] = 1.0
    k = k / k.sum()
    return k


def _depthwise_blur(x, kernel_2d):
    # x: [B, 3, H, W], kernel_2d: [K, K]
    k = kernel_2d.shape[-1]
    pad = k // 2
    w = kernel_2d.view(1, 1, k, k).repeat(3, 1, 1, 1)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_pad, w, stride=1, padding=0, groups=3)


class Dust3rWithDeblurDiNATMotionDefocus(nn.Module):
    def __init__(
        self,
        dust3r_ckpt,
        deblurdinat_repo,
        deblurdinat_weights=None,
        device="cuda",
        freeze="deblurdinat_only",
        use_grad_checkpoint=True,
        deblur_checkpoint=True,
        motion_prob=0.5,
    ):
        super().__init__()
        self.dust3r = load_model(dust3r_ckpt, device=device)
        self.deblurdinat = _load_deblurdinat(deblurdinat_repo, deblurdinat_weights)
        self.use_grad_checkpoint = use_grad_checkpoint
        self.deblur_checkpoint = deblur_checkpoint
        self.motion_prob = float(motion_prob)
        self.set_freeze(freeze)

        self.register_buffer("motion_k", _motion_kernel_25(torch.device("cpu")), persistent=False)
        self.register_buffer("defocus_k", _defocus_kernel_r7(torch.device("cpu")), persistent=False)

        if use_grad_checkpoint:
            self._enable_grad_checkpoint()

    def _enable_grad_checkpoint(self):
        def _wrap_encoder_block(blk):
            orig = blk.forward
            sig = inspect.signature(orig)
            has_kpm = "key_padding_mask" in sig.parameters

            def fwd(x, xpos, key_padding_mask=None):
                def fn(x_, xpos_):
                    if has_kpm:
                        return orig(x_, xpos_, key_padding_mask=key_padding_mask)
                    return orig(x_, xpos_)

                return checkpoint(fn, x, xpos, use_reentrant=False)

            blk.forward = fwd

        def _wrap_decoder_block(blk):
            orig = blk.forward
            sig = inspect.signature(orig)
            has_xkpm = "x_key_padding_mask" in sig.parameters
            has_ykpm = "y_key_padding_mask" in sig.parameters

            def fwd(x, y, xpos, ypos, x_key_padding_mask=None, y_key_padding_mask=None):
                def fn(x_, y_, xpos_, ypos_):
                    kwargs = {}
                    if has_xkpm:
                        kwargs["x_key_padding_mask"] = x_key_padding_mask
                    if has_ykpm:
                        kwargs["y_key_padding_mask"] = y_key_padding_mask
                    return orig(x_, y_, xpos_, ypos_, **kwargs)

                return checkpoint(fn, x, y, xpos, ypos, use_reentrant=False)

            blk.forward = fwd

        for blk in self.dust3r.enc_blocks:
            _wrap_encoder_block(blk)
        for blk in getattr(self.dust3r, "dec_blocks", []):
            _wrap_decoder_block(blk)
        for blk in getattr(self.dust3r, "dec_blocks2", []):
            _wrap_decoder_block(blk)
        print("Gradient checkpointing enabled on DUSt3R enc/dec blocks.")

    def _dust3r_decoder_modules(self):
        return (
            self.dust3r.decoder_embed,
            self.dust3r.dec_blocks,
            self.dust3r.dec_blocks2,
            self.dust3r.dec_norm,
            self.dust3r.head1,
            self.dust3r.head2,
        )

    def set_freeze(self, mode):
        valid = ("deblurdinat_only", "deblurdinat_and_decoder", "all")
        if mode not in valid:
            raise ValueError(f"freeze must be one of {valid}, got {mode!r}")

        self.freeze_mode = mode
        if mode == "deblurdinat_only":
            _set_requires_grad(self.dust3r, False)
            _set_requires_grad(self.deblurdinat, True)
        elif mode == "deblurdinat_and_decoder":
            _set_requires_grad(self.dust3r, False)
            for m in self._dust3r_decoder_modules():
                _set_requires_grad(m, True)
            _set_requires_grad(self.deblurdinat, True)
        elif mode == "all":
            _set_requires_grad(self.dust3r, True)
            _set_requires_grad(self.deblurdinat, True)

    def _apply_synthetic_blur(self, img):
        # img: [B, 3, H, W], range [-1, 1]
        x = (img + 1.0) * 0.5  # -> [0, 1]
        b = x.shape[0]
        out = []
        motion_k = self.motion_k.to(x.device)
        defocus_k = self.defocus_k.to(x.device)
        choice = torch.rand((b,), device=x.device)
        for i in range(b):
            xi = x[i : i + 1]
            if choice[i] < self.motion_prob:
                yi = _depthwise_blur(xi, motion_k)
            else:
                yi = _depthwise_blur(xi, defocus_k)
            out.append(yi)
        y = torch.cat(out, dim=0)
        y = y.clamp(0.0, 1.0)
        return y * 2.0 - 1.0  # back to [-1, 1]

    def _apply_deblur(self, view):
        v = dict(view)
        img = v["img"]  # [-1, 1]
        img_blur = self._apply_synthetic_blur(img)
        img_dinat = img_blur * 0.5  # -> [-0.5, 0.5]

        # DUSt3R training uses bf16 autocast when available; DeblurDiNAT uses
        # F.interpolate(..., nearest) which has no bf16 kernel on some CUDA builds.
        dev = img.device.type
        with torch.amp.autocast(dev, enabled=False):
            x = img_dinat.float()
            if self.deblur_checkpoint:
                def _deblur_fn(t):
                    return self.deblurdinat(t)

                deblurred = checkpoint(_deblur_fn, x, use_reentrant=False)
            else:
                deblurred = self.deblurdinat(x)

        deblurred = deblurred.to(dtype=img.dtype)
        v["img"] = deblurred * 2.0  # -> [-1, 1]
        return v

    def forward(self, view1, view2):
        view1 = self._apply_deblur(view1)
        view2 = self._apply_deblur(view2)
        return self.dust3r(view1, view2)


def build_model(
    dust3r_ckpt,
    deblurdinat_repo,
    deblurdinat_weights=None,
    device="cuda",
    freeze="deblurdinat_only",
    use_grad_checkpoint=True,
    deblur_checkpoint=True,
    motion_prob=0.5,
):
    model = Dust3rWithDeblurDiNATMotionDefocus(
        dust3r_ckpt=dust3r_ckpt,
        deblurdinat_repo=deblurdinat_repo,
        deblurdinat_weights=deblurdinat_weights,
        device=device,
        freeze=freeze,
        use_grad_checkpoint=use_grad_checkpoint,
        deblur_checkpoint=deblur_checkpoint,
        motion_prob=motion_prob,
    )
    return model.to(device)
