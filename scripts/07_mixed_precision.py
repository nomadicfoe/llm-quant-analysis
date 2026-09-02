"""Step 7: sensitivity-guided mixed precision.
Top-k most sensitive layers (by loss delta) stay INT8, the rest go INT4.
Also runs a random-k control so the sensitivity ranking is shown to matter.
Writes results/quality.csv rows variant=mixed_top{k} and mixed_random{k}."""
import random

from _common import setup
from src.data import load_codealpaca, load_mbpp
from src.evaluate import heldout_loss, mbpp_pass_at_1
from src.model import load_finetuned, load_tokenizer
from src.quant import apply_precision, avg_weight_bits, mixed, restore
from src.utils import append_csv, done_values, read_csv

cfg, log = setup("07_mixed_precision")
tok = load_tokenizer(cfg, cfg["paths"]["merged_dir"])
model = load_finetuned(cfg)
_, heldout = load_codealpaca(cfg)
problems = load_mbpp(cfg)
res_dir = cfg["paths"]["results_dir"]
group = cfg["quant"]["int4_group_size"]
n_layers = cfg["model"]["n_layers"]

sens = read_csv(f"{res_dir}/layer_sensitivity.csv")
if not sens:
    raise SystemExit("Run 05_sensitivity.py first.")
ranked = [int(r["layer"]) for r in sorted(sens, key=lambda r: -float(r["loss_delta"]))]
out = f"{res_dir}/quality.csv"
done = done_values(out, "variant")
rng = random.Random(cfg["seed"])

configs = []
for k in cfg["mixed"]["k_values"]:
    configs.append((f"mixed_top{k}", ranked[:k]))
    configs.append((f"mixed_random{k}", rng.sample(range(n_layers), k)))

for variant, int8_layers in configs:
    if variant in done:
        log.info("skip %s (done)", variant)
        continue
    plan = mixed(n_layers, int8_layers)
    orig = apply_precision(model, plan, group)
    loss = heldout_loss(model, tok, heldout, cfg)
    p1 = mbpp_pass_at_1(model, tok, problems, cfg, save_to=f"{res_dir}/generations/{variant}.jsonl")
    restore(model, orig)
    bits = avg_weight_bits(model, plan)
    log.info("%-14s int8=%s  loss %.4f  pass@1 %.4f  avg bits %.2f",
             variant, sorted(int8_layers), loss, p1, bits)
    append_csv(out, {
        "variant": variant, "stage": "finetuned", "precision": "mixed",
        "heldout_loss": round(loss, 5), "mbpp_pass1": round(p1, 4),
        "n_mbpp": len(problems), "avg_weight_bits": round(bits, 2),
        "int8_layers": " ".join(map(str, sorted(int8_layers))),
    })
