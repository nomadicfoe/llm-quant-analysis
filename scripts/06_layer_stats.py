"""Step 6: collect per-layer activation and weight statistics, correlate with sensitivity.
Writes results/layer_stats.csv and results/stat_correlations.csv."""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from _common import setup
from src.data import load_codealpaca
from src.model import load_finetuned, load_tokenizer
from src.stats import activation_stats, weight_stats
from src.utils import append_csv, read_csv

cfg, log = setup("06_layer_stats")
tok = load_tokenizer(cfg, cfg["paths"]["merged_dir"])
model = load_finetuned(cfg)
_, heldout = load_codealpaca(cfg)
res_dir = cfg["paths"]["results_dir"]

act = activation_stats(model, tok, heldout, cfg)
wts = weight_stats(model, cfg["quant"]["int4_group_size"])
sens = {int(r["layer"]): r for r in read_csv(f"{res_dir}/layer_sensitivity.csv")}
if not sens:
    raise SystemExit("Run 05_sensitivity.py first.")

stats_csv = f"{res_dir}/layer_stats.csv"
open(stats_csv, "w").close()
for a, w in zip(act, wts):
    row = {**a, **{k: v for k, v in w.items() if k != "layer"}}
    row["loss_delta"] = float(sens[a["layer"]]["loss_delta"])
    append_csv(stats_csv, row)

rows = read_csv(stats_csv)
y = np.array([float(r["loss_delta"]) for r in rows])
corr_csv = f"{res_dir}/stat_correlations.csv"
open(corr_csv, "w").close()
for feat in ["act_max_abs", "act_mean_abs", "act_var", "act_max_over_mean",
             "act_outlier_channel_frac", "w_max_abs", "w_std", "w_max_over_std",
             "recon_err_int8", "recon_err_int4"]:
    x = np.array([float(r[feat]) for r in rows])
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    log.info("%-26s pearson %+.3f (p=%.3f)  spearman %+.3f (p=%.3f)", feat, pr, pp, sr, sp)
    append_csv(corr_csv, {"feature": feat, "pearson_r": round(pr, 4), "pearson_p": round(pp, 4),
                          "spearman_r": round(sr, 4), "spearman_p": round(sp, 4)})
