#!/usr/bin/env python3
"""
Interactive Gradio demo: DUSt3R baseline, DeblurDiNAT+DUSt3R finetuned, KD Restormer, encoder LoRA.

Pipeline dropdown (each row appears only if its checkpoint path exists):
    * dust3r                         — Baseline DUSt3R (512 DPT or 224 linear via --dust3r_ckpt).
    * deblur_finetuned_5_10_20_30    — Joint DeblurDiNAT + DUSt3R, σ∈{5,10,20,30} (224).
    * deblur_finetuned_5_10_20_30_50 — Same, σ∈{5,10,20,30,50} (224).
    * kd_restormer_unet              — KD Restormer student + DUSt3R.
    * kd_encoder_lora_20ep / _50ep   — DUSt3R + encoder LoRA (dark KD).

DeblurDiNAT finetuned pipelines use --dust3r_224_ckpt and --deblur_image_size 224 automatically.

Usage (DeblurDiNAT on blurred frames, e.g. blur_s10):
    python interactive_demo/demo_finetuned.py \
        --finetuned_5_10_20_30 interactive_demo/demo_ckpts/joint_sigmas_5_10_20_30/checkpoint-best-val.pth \
        --deblurdinat_repo DeblurDiNAT \
        --image_size 224 --local_network --server_port 7860

Docker: see interactive_demo/docker/files/entrypoint.sh (env vars).
"""
from __future__ import annotations

import argparse
import functools
import inspect
import os
import os.path as osp
import sys
import tempfile
from typing import Dict, Optional

import gradio
import matplotlib.pyplot as pl
import torch

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Workaround for a gradio_client 4.x bug:
#   File ".../gradio_client/utils.py", get_type
#       if "const" in schema:        <-- crashes when schema is a bool
# It happens when Gradio builds the OpenAPI schema for components whose
# JSON schema has `additionalProperties: True/False` (e.g. dicts).  We patch
# the helpers to be bool-safe.  This keeps the UI working with gradio==4.44.x.
# ---------------------------------------------------------------------------
def _patch_gradio_client_utils():
    try:
        from gradio_client import utils as _gcu
    except Exception:
        return
    _orig_get_type = _gcu.get_type

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gcu.get_type = _safe_get_type

    _orig_json_schema = _gcu._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        try:
            return _orig_json_schema(schema, defs)
        except Exception:
            return "Any"

    _gcu._json_schema_to_python_type = _safe_json_schema_to_python_type


_patch_gradio_client_utils()

# ----------------------------------------------------------------------------
# Make sure the repo root and the dust3r submodule are importable.
# ----------------------------------------------------------------------------
_HERE = osp.dirname(osp.abspath(__file__))
_REPO = osp.dirname(_HERE)
for p in (_REPO, osp.join(_REPO, "dust3r")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _first_existing(*paths: str) -> str:
    for p in paths:
        if p and osp.isfile(p):
            return p
    return ""


def _default_dust3r_ckpt() -> str:
    """Prefer ViT-L 512 DPT (matches most finetuning/eval in this repo)."""
    env = (os.environ.get("DUST3R_CKPT") or "").strip()
    if env:
        return env if osp.isfile(env) else ""
    return _first_existing(
        osp.join(_REPO, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
        osp.join(_HERE, "demo_ckpts", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
        osp.join(_REPO, "dust3r", "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
        osp.join(_HERE, "demo_ckpts", "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"),
        osp.join(_REPO, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"),
    )


def _default_kd_restormer_student_ckpt() -> str:
    env = (os.environ.get("KD_RESTORMER_STUDENT_CKPT") or "").strip()
    if env:
        return env if osp.isfile(env) else ""
    return _first_existing(
        osp.join(
            _REPO,
            "restoration_kd_ysu",
            "outputs_from_ysu",
            "kd_restormer_frontend_1gpu",
            "student_best.pth",
        ),
        osp.join(_HERE, "demo_ckpts", "student_best.pth"),
    )


def _default_dust3r_224_ckpt() -> str:
    env = (os.environ.get("DUST3R_224_CKPT") or "").strip()
    if env:
        return env if osp.isfile(env) else ""
    return _first_existing(
        osp.join(_HERE, "demo_ckpts", "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"),
        osp.join(_REPO, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"),
        osp.join(_REPO, "dust3r", "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"),
    )


def _default_deblurdinat_repo() -> str:
    env = (os.environ.get("DEBLURDINAT_REPO") or "").strip()
    if env and osp.isdir(env):
        return env
    p = osp.join(_REPO, "DeblurDiNAT")
    return p if osp.isdir(p) else ""


def _default_finetuned_5_10_20_30() -> str:
    env = (os.environ.get("FT_5_10_20_30") or "").strip()
    if env:
        return env if osp.isfile(env) else ""
    return _first_existing(
        osp.join(_HERE, "demo_ckpts", "joint_sigmas_5_10_20_30", "checkpoint-best-val.pth"),
        osp.join(
            _REPO,
            "finetune_blur_runs",
            "deblurdinat_dust3r_asds_224_joint_5_10_20_30_from_dust3r",
            "joint_sigmas_5_10_20_30",
            "checkpoint-best-val.pth",
        ),
    )


def _default_finetuned_5_10_20_30_50() -> str:
    env = (os.environ.get("FT_5_10_20_30_50") or "").strip()
    if env:
        return env if osp.isfile(env) else ""
    return _first_existing(
        osp.join(_HERE, "demo_ckpts", "joint_sigmas_5_10_20_30_50", "checkpoint-best-val.pth"),
        osp.join(
            _REPO,
            "finetune_blur_runs",
            "deblurdinat_dust3r_asds_224_joint_5_10_20_30_50_from_dust3r",
            "joint_sigmas_5_10_20_30_50",
            "checkpoint-best-val.pth",
        ),
    )

from dust3r.demo import (  # noqa: E402
    get_3D_model_from_scene,
    get_reconstructed_scene,
    set_print_with_timestamp,
    set_scenegraph_options,
)
from dust3r.model import load_model  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402


# ----------------------------------------------------------------------------
# Pipeline metadata (keys are stable; labels are what you see in the dropdown)
# ----------------------------------------------------------------------------
PIPELINE_DUST3R = "dust3r"
PIPELINE_DEBLUR_30 = "deblur_finetuned_5_10_20_30"
PIPELINE_DEBLUR_50 = "deblur_finetuned_5_10_20_30_50"
PIPELINE_KD_RESTORMER = "kd_restormer_unet"
PIPELINE_KD20 = "kd_encoder_lora_20ep"
PIPELINE_KD50 = "kd_encoder_lora_50ep"

DEBLUR_PIPELINES = frozenset({PIPELINE_DEBLUR_30, PIPELINE_DEBLUR_50})

PIPELINE_LABELS = {
    PIPELINE_DUST3R: "Baseline (DUSt3R)",
    PIPELINE_DEBLUR_30: "DeblurDiNAT + DUSt3R finetuned (σ ∈ {5,10,20,30})",
    PIPELINE_DEBLUR_50: "DeblurDiNAT + DUSt3R finetuned (σ ∈ {5,10,20,30,50})",
    PIPELINE_KD_RESTORMER: "U-Net + DUSt3R (KD Restormer student, motion/defocus)",
    PIPELINE_KD20: "DUSt3R + encoder LoRA (KD dark, 20 epochs)",
    PIPELINE_KD50: "DUSt3R + encoder LoRA (KD dark, 50 epochs)",
}


# ----------------------------------------------------------------------------
# Model loading helpers
# ----------------------------------------------------------------------------
def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_dust3r(dust3r_ckpt: str, device: str):
    print(f"[demo] loading vanilla DUSt3R from {dust3r_ckpt}")
    model = load_model(dust3r_ckpt, device="cpu").to(device).eval()
    return model


class Dust3rWithKdRestormerFrontend(torch.nn.Module):
    """Runs StudentRestorationFrontEnd on DUSt3R-normalized RGB, then the frozen DUSt3R trunk."""

    def __init__(self, dust3r: torch.nn.Module, student: torch.nn.Module):
        super().__init__()
        self.dust3r = dust3r
        self.student = student
        self.patch_size = dust3r.patch_size
        self.square_ok = getattr(dust3r, "square_ok", False)

    def _restore(self, img_bchw: torch.Tensor) -> torch.Tensor:
        # load_images uses ImgNorm: (x01 - 0.5) / 0.5  =>  img = 2*x01 - 1
        x01 = (img_bchw + 1.0) * 0.5
        x01 = x01.clamp(0.0, 1.0)
        h, w = x01.shape[-2:]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            x01 = torch.nn.functional.pad(x01, (0, pw, 0, ph), mode="reflect")
        y01 = self.student(x01)
        y01 = y01[:, :, :h, :w].clamp(0.0, 1.0)
        return y01 * 2.0 - 1.0

    def forward(self, view1, view2):
        v1 = dict(view1)
        v2 = dict(view2)
        v1["img"] = self._restore(view1["img"])
        v2["img"] = self._restore(view2["img"])
        return self.dust3r(v1, v2)


def load_kd_restormer_student(student_ckpt: str, device: str) -> torch.nn.Module:
    from restoration_kd_ysu.train_kd_restormer_frontend import StudentRestorationFrontEnd

    print(f"[demo] loading KD Restormer student from {student_ckpt}")
    student = StudentRestorationFrontEnd().to(device).eval()
    ck = _safe_torch_load(student_ckpt)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    student.load_state_dict(state, strict=True)
    for p in student.parameters():
        p.requires_grad = False
    return student


def load_dust3r_with_kd_restormer(
    dust3r_ckpt: str,
    student_ckpt: str,
    device: str,
) -> torch.nn.Module:
    dust3r = load_model(dust3r_ckpt, device="cpu").eval().to(device)
    student = load_kd_restormer_student(student_ckpt, device)
    return Dust3rWithKdRestormerFrontend(dust3r, student).eval()


def load_deblur_finetuned(
    dust3r_224_ckpt: str,
    deblurdinat_repo: str,
    joint_ckpt: str,
    device: str,
) -> torch.nn.Module:
    from finetune_blur.deblurdinat.model import build_model

    print(f"[demo] loading DeblurDiNAT+DUSt3R joint from {joint_ckpt}")
    model = build_model(
        dust3r_ckpt=dust3r_224_ckpt,
        deblurdinat_repo=deblurdinat_repo,
        deblurdinat_weights=None,
        device="cpu",
        freeze="deblurdinat_only",
        use_grad_checkpoint=False,
        deblur_checkpoint=False,
    )
    ck = _safe_torch_load(joint_ckpt)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[demo]   missing keys: {len(missing)}")
    if unexpected:
        print(f"[demo]   unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    model.patch_size = model.dust3r.patch_size
    model.square_ok = getattr(model.dust3r, "square_ok", False)
    return model


def load_kd_encoder_lora(dust3r_ckpt: str, lora_weights: str, device: str):
    print(f"[demo] loading KD encoder LoRA from {lora_weights}")
    model = load_model(dust3r_ckpt, device="cpu")
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    ckpt = _safe_torch_load(lora_weights)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[demo]   missing keys: {len(missing)}")
    if unexpected:
        print(f"[demo]   unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    while not hasattr(inner, "patch_size") and hasattr(inner, "model"):
        inner = inner.model
    model.patch_size = inner.patch_size
    model.square_ok = getattr(inner, "square_ok", False)
    return model


# ----------------------------------------------------------------------------
# Lazy registry of loaded models.
# ----------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self._cache: Dict[str, torch.nn.Module] = {}

    def available(self) -> Dict[str, str]:
        out = {}
        base_ok = bool(self.cfg.dust3r_ckpt and osp.isfile(self.cfg.dust3r_ckpt))
        deblur_base_ok = bool(
            self.cfg.dust3r_224_ckpt
            and osp.isfile(self.cfg.dust3r_224_ckpt)
            and self.cfg.deblurdinat_repo
            and osp.isdir(self.cfg.deblurdinat_repo)
        )
        if base_ok:
            out[PIPELINE_DUST3R] = PIPELINE_LABELS[PIPELINE_DUST3R]
        if deblur_base_ok and self.cfg.finetuned_5_10_20_30 and osp.isfile(
            self.cfg.finetuned_5_10_20_30
        ):
            out[PIPELINE_DEBLUR_30] = PIPELINE_LABELS[PIPELINE_DEBLUR_30]
        if deblur_base_ok and self.cfg.finetuned_5_10_20_30_50 and osp.isfile(
            self.cfg.finetuned_5_10_20_30_50
        ):
            out[PIPELINE_DEBLUR_50] = PIPELINE_LABELS[PIPELINE_DEBLUR_50]
        if (
            base_ok
            and self.cfg.kd_restormer_student_ckpt
            and osp.isfile(self.cfg.kd_restormer_student_ckpt)
        ):
            out[PIPELINE_KD_RESTORMER] = PIPELINE_LABELS[PIPELINE_KD_RESTORMER]
        if base_ok and self.cfg.kd20_ckpt and osp.isfile(self.cfg.kd20_ckpt):
            out[PIPELINE_KD20] = PIPELINE_LABELS[PIPELINE_KD20]
        if base_ok and self.cfg.kd50_ckpt and osp.isfile(self.cfg.kd50_ckpt):
            out[PIPELINE_KD50] = PIPELINE_LABELS[PIPELINE_KD50]
        if not out:
            raise SystemExit(
                "No pipelines available. Set --dust3r_ckpt and/or DeblurDiNAT "
                "(--dust3r_224_ckpt, --deblurdinat_repo, --finetuned_5_10_20_30) "
                "and/or KD checkpoints."
            )
        return out

    def image_size_for(self, pipeline_key: str) -> int:
        if pipeline_key in DEBLUR_PIPELINES:
            return int(self.cfg.deblur_image_size)
        return int(self.cfg.image_size)

    def get(self, name: str) -> torch.nn.Module:
        if name in self._cache:
            return self._cache[name]
        device = self.cfg.device
        if name == PIPELINE_DUST3R:
            model = load_dust3r(self.cfg.dust3r_ckpt, device)
        elif name == PIPELINE_DEBLUR_30:
            model = load_deblur_finetuned(
                self.cfg.dust3r_224_ckpt,
                self.cfg.deblurdinat_repo,
                self.cfg.finetuned_5_10_20_30,
                device,
            )
        elif name == PIPELINE_DEBLUR_50:
            model = load_deblur_finetuned(
                self.cfg.dust3r_224_ckpt,
                self.cfg.deblurdinat_repo,
                self.cfg.finetuned_5_10_20_30_50,
                device,
            )
        elif name == PIPELINE_KD_RESTORMER:
            model = load_dust3r_with_kd_restormer(
                self.cfg.dust3r_ckpt,
                self.cfg.kd_restormer_student_ckpt,
                device,
            )
        elif name == PIPELINE_KD20:
            model = load_kd_encoder_lora(
                self.cfg.dust3r_ckpt,
                self.cfg.kd20_ckpt,
                device,
            )
        elif name == PIPELINE_KD50:
            model = load_kd_encoder_lora(
                self.cfg.dust3r_ckpt,
                self.cfg.kd50_ckpt,
                device,
            )
        else:
            raise ValueError(f"Unknown pipeline {name!r}")
        self._cache[name] = model
        return model


# ----------------------------------------------------------------------------
# Gradio interface (extends dust3r/demo.main_demo with a model selector).
# ----------------------------------------------------------------------------
def build_ui(tmpdir: str, registry: ModelRegistry, image_size: int,
             server_name: str, server_port: Optional[int], silent: bool):

    available = registry.available()
    pipeline_choices = [(label, key) for key, label in available.items()]
    default_pipeline = next(iter(available))

    def reconstruct(pipeline_key, *demo_args):
        model = registry.get(pipeline_key)
        sz = registry.image_size_for(pipeline_key)
        return get_reconstructed_scene(
            tmpdir, model, registry.cfg.device, silent, sz, *demo_args
        )

    refine_fn = functools.partial(get_3D_model_from_scene, tmpdir, silent)

    css = ".gradio-container {margin: 0 !important; min-width: 100%};"
    with gradio.Blocks(css=css, title="DUSt3R demo") as demo:
        scene_state = gradio.State(None)
        gradio.HTML(
            '<h2 style="text-align: center;">DUSt3R demo: baseline, DeblurDiNAT finetuned, KD</h2>'
            '<p style="text-align: center; color: #666;">'
            'DeblurDiNAT pipelines use 224px. Upload blurred frames (e.g. blur_s10), '
            'pick finetuned DeblurDiNAT+DUSt3R, then <b>Run</b>.'
            '</p>'
        )

        with gradio.Column():
            pipeline = gradio.Dropdown(
                choices=pipeline_choices,
                value=default_pipeline,
                label="Pipeline",
                info="Baseline, DeblurDiNAT+DUSt3R (Gaussian blur σ), KD Restormer, encoder LoRA",
            )

            inputfiles = gradio.File(file_count="multiple", label="Input images")

            with gradio.Row():
                schedule = gradio.Dropdown(
                    ["linear", "cosine"], value="linear",
                    label="schedule", info="Global alignment LR schedule",
                )
                niter = gradio.Number(
                    value=300, precision=0, minimum=0, maximum=5000,
                    label="num_iterations", info="Global alignment iters",
                )
                scenegraph_type = gradio.Dropdown(
                    [("complete: all possible image pairs", "complete"),
                     ("swin: sliding window", "swin"),
                     ("oneref: match one image with all", "oneref")],
                    value="complete", label="Scenegraph",
                    info="How to make pairs", interactive=True,
                )
                winsize = gradio.Slider(label="Scene Graph: Window Size",
                                        value=1, minimum=1, maximum=1, step=1,
                                        visible=False)
                refid = gradio.Slider(label="Scene Graph: Id",
                                      value=0, minimum=0, maximum=0, step=1,
                                      visible=False)

            run_btn = gradio.Button("Run", variant="primary")

            with gradio.Row():
                min_conf_thr = gradio.Slider(label="min_conf_thr",
                                             value=3.0, minimum=1.0, maximum=20, step=0.1)
                cam_size = gradio.Slider(label="cam_size",
                                         value=0.05, minimum=0.001, maximum=0.1, step=0.001)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D(label="3D reconstruction")
            outgallery = gradio.Gallery(label="rgb / depth / confidence",
                                        columns=3, height="100%")

            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])

            run_btn.click(
                fn=reconstruct,
                inputs=[pipeline, inputfiles, schedule, niter, min_conf_thr,
                        as_pointcloud, mask_sky, clean_depth, transparent_cams,
                        cam_size, scenegraph_type, winsize, refid],
                outputs=[scene_state, outmodel, outgallery],
            )

            # Live refinement of the rendered glb without re-running inference.
            for trigger in (min_conf_thr.release, cam_size.change, as_pointcloud.change,
                            mask_sky.change, clean_depth.change, transparent_cams.change):
                trigger(
                    fn=refine_fn,
                    inputs=[scene_state, min_conf_thr, as_pointcloud, mask_sky,
                            clean_depth, transparent_cams, cam_size],
                    outputs=outmodel,
                )

    # No .queue() — matches dust3r/dust3r/demo.py.
    # _frontend=False (when supported) skips Gradio's url_ok(localhost) probe, which
    # often fails inside Docker (HTTP(S)_PROXY, IPv6, race) and raises:
    #   ValueError: When localhost is not accessible...
    # It does not disable the web UI — only that pre-flight check.
    launch_kw = dict(
        share=False,
        server_name=server_name,
        server_port=server_port,
        inbrowser=False,
    )
    launch_params = inspect.signature(demo.launch).parameters
    if "_frontend" in launch_params:
        launch_kw["_frontend"] = False
    if "footer_links" in launch_params:
        launch_kw["footer_links"] = []
    demo.launch(**launch_kw)


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument(
        "--dust3r_ckpt",
        type=str,
        default=_default_dust3r_ckpt(),
        help="DUSt3R weights (default: ViT-L 512 DPT if found under checkpoints/ or demo_ckpts/)",
    )
    ap.add_argument(
        "--kd_restormer_student_ckpt",
        type=str,
        default=_default_kd_restormer_student_ckpt(),
        help="StudentRestorationFrontEnd weights (e.g. student_best.pth from kd_restormer_frontend)",
    )
    ap.add_argument("--kd20_ckpt", type=str,
                    default=os.environ.get("KD20_CKPT", ""),
                    help="KD encoder LoRA checkpoint (20-epoch run)")
    ap.add_argument("--kd50_ckpt", type=str,
                    default=os.environ.get("KD50_CKPT", ""),
                    help="KD encoder LoRA checkpoint (50-epoch run)")
    ap.add_argument(
        "--dust3r_224_ckpt",
        type=str,
        default=_default_dust3r_224_ckpt(),
        help="DUSt3R ViT-L 224 linear (DeblurDiNAT finetuned pipelines)",
    )
    ap.add_argument(
        "--deblurdinat_repo",
        type=str,
        default=_default_deblurdinat_repo(),
        help="Path to cloned DeblurDiNAT repository",
    )
    ap.add_argument(
        "--finetuned_5_10_20_30",
        type=str,
        default=_default_finetuned_5_10_20_30(),
        help="Joint checkpoint (σ ∈ {5,10,20,30})",
    )
    ap.add_argument(
        "--finetuned_5_10_20_30_50",
        type=str,
        default=_default_finetuned_5_10_20_30_50(),
        help="Joint checkpoint (σ ∈ {5,10,20,30,50})",
    )
    ap.add_argument(
        "--deblur_image_size",
        type=int,
        default=224,
        choices=[224, 512],
        help="Input size for DeblurDiNAT+DUSt3R pipelines (finetuning used 224)",
    )

    ap.add_argument(
        "--image_size",
        type=int,
        default=512,
        choices=[224, 512],
        help="Input size for baseline / KD pipelines (512 for 512_dpt)",
    )
    ap.add_argument("--device", type=str,
                    default=os.environ.get("DEVICE", "cuda"))

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--local_network", action="store_true", default=False,
                   help="Bind to 0.0.0.0 so the LAN can reach the demo")
    g.add_argument("--server_name", type=str, default=None,
                   help="Explicit bind address")
    ap.add_argument("--server_port", type=int,
                    default=int(os.environ.get("SERVER_PORT", "7860")))
    ap.add_argument("--tmp_dir", type=str, default=None)
    ap.add_argument("--silent", action="store_true", default=False)
    return ap.parse_args()


def resolve_device(requested: str) -> str:
    """Use CPU when CUDA is requested but no GPU is visible (e.g. login node / mis-set CUDA_VISIBLE_DEVICES)."""
    r = (requested or "cuda").strip()
    if r.lower() == "cpu":
        return "cpu"
    if r.lower().startswith("cuda"):
        if not torch.cuda.is_available():
            print(
                "[demo] WARNING: device=cuda but no CUDA GPU is available; "
                "using CPU (much slower). Pass --device cpu to silence this, "
                "or run on a machine with a visible GPU."
            )
            return "cpu"
        return r
    return r


def main():
    args = get_args()
    args.device = resolve_device(args.device)
    set_print_with_timestamp()

    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
        tempfile.tempdir = args.tmp_dir

    server_name = args.server_name or ("0.0.0.0" if args.local_network else "127.0.0.1")

    registry = ModelRegistry(args)
    avail = registry.available()
    print("[demo] available pipelines:")
    for k, label in avail.items():
        print(f"  - {k}: {label}")

    with tempfile.TemporaryDirectory(suffix="_dust3r_finetuned_demo") as tmpdir:
        if not args.silent:
            print(f"[demo] tmp dir: {tmpdir}")
        build_ui(tmpdir, registry, args.image_size,
                 server_name, args.server_port, args.silent)


if __name__ == "__main__":
    main()
