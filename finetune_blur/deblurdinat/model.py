"""
DUSt3R + DeblurDiNAT wrapper for finetuning.

Pipeline:  blurred_img --> [DeblurDiNAT] --> deblurred_img --> DUSt3R --> (pts3d, conf)

DeblurDiNAT-L (NADeblurPlus) is loaded from the DeblurDiNAT repo with pretrained
GoPro weights, then finetuned end-to-end with DUSt3R's ConfLoss.
"""

import sys
import os.path as osp
import inspect
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../../dust3r')))
from dust3r.model import load_model


def _set_requires_grad(module, requires_grad):
    if not hasattr(module, 'parameters'):
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


def _deblurdinat_state_dict_from_checkpoint(ckpt):
    """Handle nested checkpoints and DDP/DataParallel ``module.`` key prefix."""
    state = ckpt
    if isinstance(state, dict):
        for k in ('state_dict', 'model', 'params'):
            inner = state.get(k)
            if isinstance(inner, dict) and inner and all(isinstance(x, str) for x in inner):
                state = inner
                break
    if not isinstance(state, dict):
        raise ValueError('DeblurDiNAT weights must be a state_dict or dict containing one')
    out = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith('module.'):
            out[k[len('module.'):]] = v
        else:
            out[k] = v
    return out


def _load_deblurdinat(deblurdinat_repo, weights_path=None):
    """Import NADeblurPlus from the DeblurDiNAT repo and optionally load weights."""
    models_dir = osp.join(deblurdinat_repo, 'models')
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    if deblurdinat_repo not in sys.path:
        sys.path.insert(0, deblurdinat_repo)

    from DeblurDiNATL import NADeblurPlus
    net = NADeblurPlus()

    if weights_path and osp.isfile(weights_path):
        try:
            raw = torch.load(weights_path, map_location='cpu', weights_only=False)
        except TypeError:
            raw = torch.load(weights_path, map_location='cpu')
        state = _deblurdinat_state_dict_from_checkpoint(raw)
        # Older DeblurDiNAT checkpoints store na2d.rpb (relative position bias).
        # NATTEN >= 0.21 NeighborhoodAttention2D has no rpb; drop those keys.
        n_before = len(state)
        state = {k: v for k, v in state.items() if '.rpb' not in k}
        n_drop = n_before - len(state)
        if n_drop:
            print(f'  (omitted {n_drop} .rpb keys — incompatible with installed NATTEN)')
        net.load_state_dict(state, strict=True)
        print(f'Loaded DeblurDiNAT weights from {weights_path}')

    return net


class Dust3rWithDeblurDiNAT(nn.Module):
    """
    Wraps pretrained DUSt3R with DeblurDiNAT-L as a learned deblurring front-end.

    Args:
        dust3r_ckpt      : path to pretrained DUSt3R checkpoint
        deblurdinat_repo : path to cloned DeblurDiNAT repository
        deblurdinat_weights : path to DeblurDiNATL.pth (GoPro pretrained)
        device           : 'cpu' or 'cuda'
        freeze           : 'deblurdinat_only' | 'deblurdinat_and_decoder' | 'all'
    """

    def __init__(self, dust3r_ckpt, deblurdinat_repo, deblurdinat_weights=None,
                 device='cuda', freeze='deblurdinat_only',
                 use_grad_checkpoint=True, deblur_checkpoint=True):
        super().__init__()
        self.dust3r = load_model(dust3r_ckpt, device=device)
        self.deblurdinat = _load_deblurdinat(deblurdinat_repo, deblurdinat_weights)
        self.use_grad_checkpoint = use_grad_checkpoint
        self.deblur_checkpoint = deblur_checkpoint
        self.set_freeze(freeze)
        if use_grad_checkpoint:
            self._enable_grad_checkpoint()

    def _enable_grad_checkpoint(self):
        """
        Wrap DUSt3R encoder ``Block`` and decoder ``DecoderBlock`` forwards with
        ``torch.utils.checkpoint``.  Masks are captured in a closure (not
        checkpoint inputs).  Decoder returns ``(x, y)`` — supported by checkpoint.
        """
        def _wrap_encoder_block(blk):
            orig = blk.forward
            sig = inspect.signature(orig)
            has_kpm = 'key_padding_mask' in sig.parameters

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
            has_xkpm = 'x_key_padding_mask' in sig.parameters
            has_ykpm = 'y_key_padding_mask' in sig.parameters

            def fwd(x, y, xpos, ypos, x_key_padding_mask=None, y_key_padding_mask=None):
                def fn(x_, y_, xpos_, ypos_):
                    kwargs = {}
                    if has_xkpm:
                        kwargs['x_key_padding_mask'] = x_key_padding_mask
                    if has_ykpm:
                        kwargs['y_key_padding_mask'] = y_key_padding_mask
                    return orig(x_, y_, xpos_, ypos_, **kwargs)

                return checkpoint(fn, x, y, xpos, ypos, use_reentrant=False)

            blk.forward = fwd

        for blk in self.dust3r.enc_blocks:
            _wrap_encoder_block(blk)
        for blk in getattr(self.dust3r, 'dec_blocks', []):
            _wrap_decoder_block(blk)
        for blk in getattr(self.dust3r, 'dec_blocks2', []):
            _wrap_decoder_block(blk)
        print('Gradient checkpointing enabled on DUSt3R enc/dec blocks.')

    def _dust3r_encoder_modules(self):
        return (
            self.dust3r.patch_embed,
            self.dust3r.enc_blocks,
            self.dust3r.enc_norm,
        )

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
        """
        'deblurdinat_only'        : freeze DUSt3R, train only DeblurDiNAT
        'deblurdinat_and_decoder' : freeze encoder, train DeblurDiNAT + decoder + heads
        'all'                     : train everything end-to-end
        """
        valid = ('deblurdinat_only', 'deblurdinat_and_decoder', 'all')
        if mode not in valid:
            raise ValueError(f'freeze must be one of {valid}, got {mode!r}')

        self.freeze_mode = mode

        if mode == 'deblurdinat_only':
            _set_requires_grad(self.dust3r, False)
            _set_requires_grad(self.deblurdinat, True)

        elif mode == 'deblurdinat_and_decoder':
            _set_requires_grad(self.dust3r, False)
            for m in self._dust3r_decoder_modules():
                _set_requires_grad(m, True)
            _set_requires_grad(self.deblurdinat, True)

        elif mode == 'all':
            _set_requires_grad(self.dust3r, True)
            _set_requires_grad(self.deblurdinat, True)

    def _apply_deblur(self, view):
        """
        DeblurDiNAT expects input in [-0.5, 0.5] and outputs a residual (out + input).
        DUSt3R uses [-1, 1].  Convert accordingly.

        Optional ``torch.utils.checkpoint`` on the whole DeblurDiNAT forward avoids
        storing full NATTEN activations (major memory at 512²); recomputed in backward.
        """
        v = dict(view)
        img = v['img']                           # (B, 3, H, W) in [-1, 1]
        img_dinat = img * 0.5                    # -> [-0.5, 0.5]
        if self.deblur_checkpoint:
            def _deblur_fn(x):
                return self.deblurdinat(x)

            deblurred = checkpoint(_deblur_fn, img_dinat, use_reentrant=False)
        else:
            deblurred = self.deblurdinat(img_dinat)
        v['img'] = deblurred * 2.0               # -> [-1, 1]
        return v

    def forward(self, view1, view2):
        view1 = self._apply_deblur(view1)
        view2 = self._apply_deblur(view2)
        return self.dust3r(view1, view2)


def build_model(dust3r_ckpt, deblurdinat_repo, deblurdinat_weights=None,
                device='cuda', freeze='deblurdinat_only',
                use_grad_checkpoint=True, deblur_checkpoint=True):
    model = Dust3rWithDeblurDiNAT(
        dust3r_ckpt=dust3r_ckpt,
        deblurdinat_repo=deblurdinat_repo,
        deblurdinat_weights=deblurdinat_weights,
        device=device,
        freeze=freeze,
        use_grad_checkpoint=use_grad_checkpoint,
        deblur_checkpoint=deblur_checkpoint,
    )
    return model.to(device)
