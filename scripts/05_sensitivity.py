"""Step 5: layer-wise INT4 sensitivity sweep.
For each decoder layer i: layer i -> INT4, everything else FP16. Measure held-out loss
delta (all layers), then MBPP pass@1 on the most sensitive layers (or all, if configured).
Writes results/layer_sensitivity.csv. Resumable."""
from _common import setup
from src.data import load_codealpaca, load_mbpp
from src.evaluate import heldout_loss, mbpp_pass_at_1
from src.model import load_finetuned, load_tokenizer
from src.quant import apply_precision, restore
from src.utils import append_csv, read_csv

cfg, log = setup("05_sensitivity")
tok = load_tokenizer(cfg, cfg["paths"]["merged_dir"])
model = load_finetuned(cfg)
_, heldout = load_codealpaca(cfg)
problems = load_mbpp(cfg)
group = cfg["quant"]["int4_group_size"]
n_layers = cfg["model"]["n_layers"]
res_dir = cfg["paths"]["results_dir"]

quality = {r["variant"]: r for r in read_csv(f"{res_dir}/quality.csv")}
if "finetuned_fp16" not in quality:
    raise SystemExit("Run 02_finetune.py first (need finetuned_fp16 reference).")
ref_loss = float(quality["finetuned_fp16"]["heldout_loss"])
ref_p1 = float(quality["finetuned_fp16"]["mbpp_pass1"])

# Pass A: loss for every layer (cheap)
loss_csv = f"{res_dir}/layer_sensitivity_loss.csv"
done = {int(r["layer"]) for r in read_csv(loss_csv)}
for i in range(n_layers):
    if i in done:
        continue
    orig = apply_precision(model, {i: "int4"}, group)
    loss = heldout_loss(model, tok, heldout, cfg)
    restore(model, orig)
    log.info("layer %2d int4  loss %.4f  delta %+.4f", i, loss, loss - ref_loss)
    append_csv(loss_csv, {"layer": i, "heldout_loss": round(loss, 5),
                          "loss_delta": round(loss - ref_loss, 5)})

# Pass B: pass@1 for selected layers (expensive)
rows = sorted(read_csv(loss_csv), key=lambda r: -float(r["loss_delta"]))
if cfg["sensitivity"]["with_pass1"]:
    targets = [int(r["layer"]) for r in rows]
else:
    targets = [int(r["layer"]) for r in rows[: cfg["sensitivity"]["n_pass1_layers"]]]

p1_csv = f"{res_dir}/layer_sensitivity_pass1.csv"
done = {int(r["layer"]) for r in read_csv(p1_csv)}
for i in targets:
    if i in done:
        continue
    orig = apply_precision(model, {i: "int4"}, group)
    p1 = mbpp_pass_at_1(model, tok, problems, cfg)
    restore(model, orig)
    log.info("layer %2d int4  pass@1 %.4f  drop %+.4f", i, p1, ref_p1 - p1)
    append_csv(p1_csv, {"layer": i, "mbpp_pass1": round(p1, 4), "pass1_drop": round(ref_p1 - p1, 4)})

# Merge into one table
p1_map = {int(r["layer"]): r for r in read_csv(p1_csv)}
merged_csv = f"{res_dir}/layer_sensitivity.csv"
open(merged_csv, "w").close()
for r in sorted(read_csv(loss_csv), key=lambda r: int(r["layer"])):
    i = int(r["layer"])
    append_csv(merged_csv, {
        "layer": i, "heldout_loss": r["heldout_loss"], "loss_delta": r["loss_delta"],
        "mbpp_pass1": p1_map.get(i, {}).get("mbpp_pass1", ""),
        "pass1_drop": p1_map.get(i, {}).get("pass1_drop", ""),
    })
log.info("wrote %s", merged_csv)
