"""Step 4: real-kernel efficiency (bitsandbytes) for FP16 / INT8 / NF4.
Writes results/efficiency.csv. Each precision is loaded fresh so VRAM numbers are clean."""
import gc

import torch

from _common import setup
from src.benchmark import load_for_benchmark, measure
from src.model import load_tokenizer
from src.utils import append_csv, done_values

cfg, log = setup("04_benchmark")
tok = load_tokenizer(cfg, cfg["paths"]["merged_dir"])
out = f"{cfg['paths']['results_dir']}/efficiency.csv"
done = done_values(out, "precision")

for prec in ("fp16", "int8", "int4"):
    if prec in done:
        log.info("skip %s (done)", prec)
        continue
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_for_benchmark(cfg, prec)
    row = measure(model, tok, cfg, prec)
    log.info("%s", row)
    append_csv(out, row)
    del model
    gc.collect()
    torch.cuda.empty_cache()
