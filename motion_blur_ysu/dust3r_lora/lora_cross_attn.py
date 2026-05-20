"""Inject PEFT LoRA into CrossAttention Q/K/V linears only (decoder cross-view)."""
from dust3r.model import load_model


def build_lora_dust3r_cross_attn(
    dust3r_ckpt: str,
    device: str = "cpu",
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("Install peft: pip install 'peft>=0.11'") from e

    model = load_model(dust3r_ckpt, device=device, verbose=False)
    for p in model.parameters():
        p.requires_grad = False

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["projq", "projk", "projv"],
        modules_to_save=None,
    )
    model = get_peft_model(model, cfg)
    return model


def count_trainable(model):
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    return n_t, n_all
