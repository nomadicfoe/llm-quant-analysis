"""Inference efficiency with real quantized kernels (bitsandbytes).

Measures per precision: model size on GPU, peak VRAM during generation,
first-token latency, and decode throughput. These are the numbers simulated
quantization cannot give you.
"""
import time

import torch
from transformers import BitsAndBytesConfig

from .model import load_model


def bnb_config(precision: str):
    if precision == "fp16":
        return None
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "int4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=False)
    raise ValueError(precision)


def load_for_benchmark(cfg: dict, precision: str):
    return load_model(cfg, dtype="float16", path=cfg["paths"]["merged_dir"],
                      quantization_config=bnb_config(precision))


@torch.no_grad()
def measure(model, tok, cfg: dict, precision: str) -> dict:
    b = cfg["benchmark"]
    enc = tok(b["prompt"], return_tensors="pt").to(model.device)
    common = dict(do_sample=False, pad_token_id=tok.pad_token_id)

    for _ in range(b["n_warmup"]):
        model.generate(**enc, max_new_tokens=8, **common)
    torch.cuda.synchronize()

    # first-token latency
    ttft = []
    for _ in range(b["n_runs"]):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**enc, max_new_tokens=1, **common)
        torch.cuda.synchronize()
        ttft.append(time.perf_counter() - t0)

    # decode throughput
    torch.cuda.reset_peak_memory_stats()
    tps, total = [], []
    n = b["gen_tokens"]
    for _ in range(b["n_runs"]):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**enc, max_new_tokens=n, min_new_tokens=n, **common)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out.shape[1] - enc["input_ids"].shape[1]
        tps.append(gen / dt)
        total.append(dt)

    return {
        "precision": precision,
        "model_size_mb": round(model.get_memory_footprint() / 2**20, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "first_token_ms": round(1000 * sum(ttft) / len(ttft), 2),
        "gen_latency_ms": round(1000 * sum(total) / len(total), 1),
        "tokens_per_sec": round(sum(tps) / len(tps), 2),
        "gen_tokens": n,
        "gpu": torch.cuda.get_device_name(0),
    }
