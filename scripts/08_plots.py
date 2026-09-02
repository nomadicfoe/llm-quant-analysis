"""Step 8: all figures from the CSVs. No model needed, runs on CPU in seconds."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import setup
from src.utils import read_csv

cfg, log = setup("08_plots")
res, fig_dir = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]


def save(name):
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{name}.png", dpi=150)
    plt.close()
    log.info("wrote %s/%s.png", fig_dir, name)


# 1. quality by variant
q = read_csv(f"{res}/quality.csv")
if q:
    order = ["base_fp16", "finetuned_fp16", "finetuned_int8", "finetuned_int4"] + \
            [r["variant"] for r in q if r["variant"].startswith("mixed")]
    rows = [next(r for r in q if r["variant"] == v) for v in order if any(r["variant"] == v for r in q)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = [r["variant"].replace("finetuned_", "ft_") for r in rows]
    ax[0].bar(names, [100 * float(r["mbpp_pass1"]) for r in rows], color="#4C72B0")
    ax[0].set_ylabel("MBPP pass@1 (%)"); ax[0].set_title("Task quality")
    ax[0].tick_params(axis="x", rotation=35)
    ax[1].bar(names, [float(r["heldout_loss"]) for r in rows], color="#DD8452")
    ax[1].set_ylabel("held-out loss (nats/token)"); ax[1].set_title("Held-out loss (lower is better)")
    ax[1].tick_params(axis="x", rotation=35)
    save("quality_vs_precision")

    # quality vs bits tradeoff
    ft = [r for r in rows if r["variant"] != "base_fp16"]
    plt.figure(figsize=(6, 4.5))
    for r in ft:
        plt.scatter(float(r["avg_weight_bits"]), 100 * float(r["mbpp_pass1"]), s=70)
        plt.annotate(r["variant"].replace("finetuned_", ""),
                     (float(r["avg_weight_bits"]), 100 * float(r["mbpp_pass1"])),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("average bits per weight (decoder blocks)"); plt.ylabel("MBPP pass@1 (%)")
    plt.title("Quality vs compression"); plt.grid(alpha=0.3)
    save("quality_vs_bits")

# 2. efficiency
e = read_csv(f"{res}/efficiency.csv")
if e:
    order = {"fp16": 0, "int8": 1, "int4": 2}
    e = sorted(e, key=lambda r: order.get(r["precision"], 9))
    names = [r["precision"].upper() for r in e]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, col, title in zip(ax, ["model_size_mb", "peak_vram_mb", "first_token_ms", "tokens_per_sec"],
                             ["Model size (MB)", "Peak VRAM (MB)", "First-token latency (ms)", "Tokens / sec"]):
        a.bar(names, [float(r[col]) for r in e], color="#55A868")
        a.set_title(title)
    fig.suptitle(f"Real-kernel inference efficiency (bitsandbytes, {e[0]['gpu']})")
    save("efficiency_vs_precision")

# 3. layer sensitivity
s = read_csv(f"{res}/layer_sensitivity.csv")
if s:
    layers = [int(r["layer"]) for r in s]
    d = [float(r["loss_delta"]) for r in s]
    plt.figure(figsize=(10, 4))
    plt.bar(layers, d, color="#C44E52")
    plt.xlabel("transformer layer quantized to INT4 (others FP16)")
    plt.ylabel("held-out loss increase vs FP16")
    plt.title("Layer-wise INT4 quantization sensitivity")
    plt.xticks(layers)
    save("layer_sensitivity")
    p = [(int(r["layer"]), float(r["pass1_drop"])) for r in s if r["pass1_drop"]]
    if p:
        plt.figure(figsize=(10, 4))
        plt.bar([x for x, _ in p], [100 * y for _, y in p], color="#8172B2")
        plt.xlabel("layer (INT4, others FP16)"); plt.ylabel("MBPP pass@1 drop (pts)")
        plt.title("Task-level sensitivity for most loss-sensitive layers")
        save("layer_sensitivity_pass1")

# 4. stats vs sensitivity
st = read_csv(f"{res}/layer_stats.csv")
if st:
    feats = [("act_max_abs", "max |activation| into layer"),
             ("act_outlier_channel_frac", "outlier channel fraction (>6)"),
             ("recon_err_int4", "INT4 weight reconstruction error"),
             ("w_max_over_std", "weight max / std")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    y = [float(r["loss_delta"]) for r in st]
    for a, (f, lab) in zip(ax, feats):
        x = [float(r[f]) for r in st]
        a.scatter(x, y, s=30)
        for r, xi, yi in zip(st, x, y):
            a.annotate(r["layer"], (xi, yi), fontsize=6, textcoords="offset points", xytext=(2, 2))
        a.set_xlabel(lab); a.set_ylabel("loss delta (INT4)")
    fig.suptitle("Does any layer statistic predict INT4 sensitivity?")
    save("stats_vs_sensitivity")

    c = read_csv(f"{res}/stat_correlations.csv")
    if c:
        plt.figure(figsize=(8, 4))
        plt.barh([r["feature"] for r in c], [float(r["spearman_r"]) for r in c], color="#64B5CD")
        plt.axvline(0, color="k", lw=0.8)
        plt.xlabel("Spearman correlation with INT4 loss delta")
        plt.title("Layer statistics vs sensitivity")
        save("stat_correlations")
