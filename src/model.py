"""Model / tokenizer loading and transformer-layer accessors."""
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import device

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def load_tokenizer(cfg: dict, path: str | None = None):
    tok = AutoTokenizer.from_pretrained(path or cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # batched generation
    return tok


def load_model(cfg: dict, dtype: str = "float16", path: str | None = None,
               quantization_config=None):
    """Load base model (path=None) or a merged checkpoint (path=merged_dir)."""
    kwargs = dict(torch_dtype=DTYPES[dtype])
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path or cfg["model"]["name"], **kwargs)
    if quantization_config is None:
        model.to(device())
    model.eval()
    return model


def merged_exists(cfg: dict) -> bool:
    return (Path(cfg["paths"]["merged_dir"]) / "config.json").exists()


def load_finetuned(cfg: dict, dtype: str = "float16"):
    if not merged_exists(cfg):
        raise FileNotFoundError("No merged model. Run scripts/02_finetune.py first.")
    return load_model(cfg, dtype=dtype, path=cfg["paths"]["merged_dir"])


def decoder_layers(model) -> nn.ModuleList:
    """Works for Qwen2 / Llama style models: model.model.layers."""
    return model.model.layers


def linear_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    """All nn.Linear modules inside one decoder block, keyed by qualified name."""
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
