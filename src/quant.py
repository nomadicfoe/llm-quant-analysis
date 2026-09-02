"""Simulated (fake) weight quantization with per-layer precision control.

Weights are quantized then dequantized in place, so the model runs in FP16 but
with the exact rounding error a real INT8/INT4 model would have. This is the
standard way quantization papers measure quality, and it lets us set any
precision on any layer, which bitsandbytes cannot do cleanly.

INT8: symmetric, per output channel.
INT4: asymmetric, group-wise along the input dimension (group 128, like GPTQ / NF4 grouping).

Real-kernel memory and latency numbers come from benchmark.py, not from here.
"""
import torch
from torch import nn

from .model import decoder_layers, linear_modules

PRECISIONS = ("fp16", "int8", "int4")


@torch.no_grad()
def fake_quant_int8(w: torch.Tensor) -> torch.Tensor:
    w32 = w.float()
    scale = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(w32 / scale).clamp(-128, 127)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def fake_quant_int4(w: torch.Tensor, group: int = 128) -> torch.Tensor:
    out_f, in_f = w.shape
    w32 = w.float()
    pad = (-in_f) % group
    if pad:
        w32 = torch.nn.functional.pad(w32, (0, pad))
    g = w32.view(out_f, -1, group)
    mn = g.amin(dim=2, keepdim=True)
    mx = g.amax(dim=2, keepdim=True)
    scale = ((mx - mn) / 15.0).clamp(min=1e-8)
    zero = torch.round(-mn / scale)
    q = torch.clamp(torch.round(g / scale) + zero, 0, 15)
    deq = ((q - zero) * scale).view(out_f, -1)[:, :in_f]
    return deq.to(w.dtype)


def fake_quant(w: torch.Tensor, precision: str, group: int = 128) -> torch.Tensor:
    if precision == "fp16":
        return w
    if precision == "int8":
        return fake_quant_int8(w)
    if precision == "int4":
        return fake_quant_int4(w, group)
    raise ValueError(precision)


@torch.no_grad()
def recon_error(w: torch.Tensor, precision: str, group: int = 128) -> float:
    """Relative Frobenius reconstruction error ||w - q(w)|| / ||w||."""
    if precision == "fp16":
        return 0.0
    w32 = w.float()
    return (torch.norm(w32 - fake_quant(w, precision, group).float()) / torch.norm(w32)).item()


@torch.no_grad()
def apply_precision(model, layer_precisions: dict[int, str], group: int = 128) -> dict:
    """Quantize the given layers in place. Returns originals for restore()."""
    layers = decoder_layers(model)
    originals = {}
    for idx, prec in layer_precisions.items():
        if prec == "fp16":
            continue
        for name, lin in linear_modules(layers[idx]).items():
            originals[(idx, name)] = lin.weight.data.clone()
            lin.weight.data.copy_(fake_quant(lin.weight.data, prec, group))
    return originals


@torch.no_grad()
def restore(model, originals: dict) -> None:
    layers = decoder_layers(model)
    for (idx, name), w in originals.items():
        linear_modules(layers[idx])[name].weight.data.copy_(w)


def uniform(n_layers: int, precision: str) -> dict[int, str]:
    return {i: precision for i in range(n_layers)}


def mixed(n_layers: int, int8_layers: list[int]) -> dict[int, str]:
    return {i: ("int8" if i in int8_layers else "int4") for i in range(n_layers)}


def avg_weight_bits(model, layer_precisions: dict[int, str]) -> float:
    """Average bits per weight over the decoder-block Linear weights only
    (embeddings / lm_head / norms stay FP16 and are excluded)."""
    bits = {"fp16": 16, "int8": 8, "int4": 4}
    total, weighted = 0, 0
    for idx, layer in enumerate(decoder_layers(model)):
        n = sum(m.weight.numel() for m in linear_modules(layer).values())
        total += n
        weighted += n * bits[layer_precisions.get(idx, "fp16")]
    return weighted / total
