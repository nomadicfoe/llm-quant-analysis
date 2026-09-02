"""Per-layer statistics used to explain quantization sensitivity.

Activation stats come from forward pre-hooks on each decoder block (the hidden
state entering the block). Weight stats and reconstruction errors come straight
from the Linear weights.
"""
import torch

from .model import decoder_layers, linear_modules
from .quant import recon_error


@torch.no_grad()
def activation_stats(model, tok, examples: list[dict], cfg: dict, n_examples: int = 128) -> list[dict]:
    layers = decoder_layers(model)
    acc = [{"max_abs": 0.0, "sum": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "n": 0, "outlier_frac_sum": 0.0, "batches": 0}
           for _ in layers]
    handles = []

    def make_hook(i):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            h = h.detach().float()
            a = acc[i]
            a["max_abs"] = max(a["max_abs"], h.abs().max().item())
            a["sum"] += h.sum().item()
            a["sum_abs"] += h.abs().sum().item()
            a["sum_sq"] += (h ** 2).sum().item()
            a["n"] += h.numel()
            # fraction of channels whose max activation exceeds 6 (LLM.int8 outlier threshold)
            ch_max = h.abs().reshape(-1, h.shape[-1]).amax(dim=0)
            a["outlier_frac_sum"] += (ch_max > 6.0).float().mean().item()
            a["batches"] += 1
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_hook(i), with_kwargs=True))

    tok.padding_side = "right"
    bs = cfg["eval"]["loss_batch_size"]
    subset = examples[:n_examples]
    for s in range(0, len(subset), bs):
        texts = [ex["prompt"] + ex["response"] for ex in subset[s:s + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg["data"]["max_seq_len"]).to(model.device)
        model(**enc)
    tok.padding_side = "left"
    for h in handles:
        h.remove()

    rows = []
    for i, a in enumerate(acc):
        mean_abs = a["sum_abs"] / a["n"]
        mean = a["sum"] / a["n"]
        var = a["sum_sq"] / a["n"] - mean ** 2
        rows.append({
            "layer": i,
            "act_max_abs": round(a["max_abs"], 4),
            "act_mean_abs": round(mean_abs, 6),
            "act_var": round(var, 6),
            "act_max_over_mean": round(a["max_abs"] / max(mean_abs, 1e-8), 2),
            "act_outlier_channel_frac": round(a["outlier_frac_sum"] / a["batches"], 5),
        })
    return rows


@torch.no_grad()
def weight_stats(model, group: int) -> list[dict]:
    rows = []
    for i, layer in enumerate(decoder_layers(model)):
        lins = linear_modules(layer)
        w_max = max(m.weight.abs().max().item() for m in lins.values())
        numel = sum(m.weight.numel() for m in lins.values())
        w_std = (sum((m.weight.float() ** 2).sum().item() for m in lins.values()) / numel) ** 0.5
        # parameter-weighted mean relative reconstruction error
        e8 = sum(recon_error(m.weight, "int8", group) * m.weight.numel() for m in lins.values()) / numel
        e4 = sum(recon_error(m.weight, "int4", group) * m.weight.numel() for m in lins.values()) / numel
        rows.append({
            "layer": i,
            "w_max_abs": round(w_max, 5),
            "w_std": round(w_std, 6),
            "w_max_over_std": round(w_max / w_std, 2),
            "recon_err_int8": round(e8, 6),
            "recon_err_int4": round(e4, 6),
        })
    return rows
