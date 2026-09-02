"""Step 1: evaluate the pretrained base model before any fine-tuning.
Writes results/quality.csv row variant=base_fp16."""
from _common import setup
from src.data import load_codealpaca, load_mbpp
from src.evaluate import heldout_loss, mbpp_pass_at_1
from src.model import load_model, load_tokenizer, n_params
from src.utils import append_csv

cfg, log = setup("01_baseline")
tok = load_tokenizer(cfg)
model = load_model(cfg, dtype="float16")
log.info("loaded %s with %.1fM params", cfg["model"]["name"], n_params(model) / 1e6)

_, heldout = load_codealpaca(cfg)
problems = load_mbpp(cfg)

loss = heldout_loss(model, tok, heldout, cfg)
log.info("base heldout loss %.4f", loss)
p1 = mbpp_pass_at_1(model, tok, problems, cfg,
                    save_to=f"{cfg['paths']['results_dir']}/generations/base_fp16.jsonl")
log.info("base MBPP pass@1 %.4f", p1)

append_csv(f"{cfg['paths']['results_dir']}/quality.csv", {
    "variant": "base_fp16", "stage": "pretrained", "precision": "fp16",
    "heldout_loss": round(loss, 5), "mbpp_pass1": round(p1, 4),
    "n_mbpp": len(problems), "avg_weight_bits": 16.0,
})
