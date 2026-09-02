"""Step 3: uniform simulated INT8 and INT4 on the fine-tuned model. Quality only.
Writes results/quality.csv rows variant=finetuned_int8 / finetuned_int4."""
from _common import setup
from src.data import load_codealpaca, load_mbpp
from src.evaluate import heldout_loss, mbpp_pass_at_1
from src.model import load_finetuned, load_tokenizer
from src.quant import apply_precision, avg_weight_bits, restore, uniform
from src.utils import append_csv, done_values

cfg, log = setup("03_quantize_eval")
tok = load_tokenizer(cfg, cfg["paths"]["merged_dir"])
model = load_finetuned(cfg)
_, heldout = load_codealpaca(cfg)
problems = load_mbpp(cfg)
out = f"{cfg['paths']['results_dir']}/quality.csv"
done = done_values(out, "variant")
n_layers = cfg["model"]["n_layers"]
group = cfg["quant"]["int4_group_size"]

for prec in ("int8", "int4"):
    variant = f"finetuned_{prec}"
    if variant in done:
        log.info("skip %s (done)", variant)
        continue
    plan = uniform(n_layers, prec)
    orig = apply_precision(model, plan, group)
    loss = heldout_loss(model, tok, heldout, cfg)
    p1 = mbpp_pass_at_1(model, tok, problems, cfg,
                        save_to=f"{cfg['paths']['results_dir']}/generations/{variant}.jsonl")
    restore(model, orig)
    log.info("%s  loss %.4f  pass@1 %.4f", variant, loss, p1)
    append_csv(out, {
        "variant": variant, "stage": "finetuned", "precision": prec,
        "heldout_loss": round(loss, 5), "mbpp_pass1": round(p1, 4),
        "n_mbpp": len(problems), "avg_weight_bits": round(avg_weight_bits(model, plan), 2),
    })
