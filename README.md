# Layer-Adaptive Low-Bit Quantization for Small Language Models

Controlled study of what happens when a small LLM is LoRA fine-tuned for code generation and then compressed to INT8 / INT4 for inference.

**Question:** How much task quality is lost going FP16 → INT8 → INT4, which transformer layers are responsible for that loss, and can a sensitivity-guided mixed-precision layout recover it while keeping most of INT4's memory savings?

> Results section below is filled in after running the pipeline. See `RUNBOOK.md` for the exact procedure.

## Setup

| | |
|---|---|
| Model | Qwen2.5-0.5B (base), 24 decoder layers |
| Fine-tuning | LoRA r=16 on 4,000 Python examples from CodeAlpaca-20k, 2 epochs |
| Task metric | MBPP test (500 problems), pass@1 by executing generated code against unit tests |
| Proxy metric | Held-out loss on 500 unseen CodeAlpaca examples (response tokens only) |
| Quality quantization | Simulated PTQ: INT8 per-channel symmetric, INT4 group-128 asymmetric, any layer set |
| Efficiency quantization | Real kernels via bitsandbytes (LLM.int8, NF4) |
| Hardware | Kaggle T4 |

Two quantization tracks on purpose. Simulated quantization gives exact per-layer control needed for the sensitivity sweep and mixed model; real kernels give honest memory and latency numbers. Simulated quant has no speed benefit and real kernels can't mix precisions per layer, so each is used for what it does well.

## Pipeline

```
01_baseline        base model quality
02_finetune        LoRA -> merge -> FP16 checkpoint -> quality
03_quantize_eval   uniform INT8 / INT4 quality
04_benchmark       VRAM, size, first-token latency, tokens/sec (bitsandbytes)
05_sensitivity     one layer at a time -> INT4, rest FP16, measure degradation
06_layer_stats     activation / weight statistics per layer, correlate with sensitivity
07_mixed_precision top-k sensitive layers INT8, rest INT4, plus random-k control
08_plots           all figures from CSVs
```

## Results

<!-- Fill in from results/quality.csv and results/efficiency.csv -->

### Quality vs precision

| Variant | Avg weight bits | Held-out loss | MBPP pass@1 |
|---|---|---|---|
| Base FP16 | 16 | | |
| Fine-tuned FP16 | 16 | | |
| Fine-tuned INT8 | 8 | | |
| Fine-tuned INT4 | 4 | | |
| Mixed top-4 (INT8) / INT4 | | | |
| Mixed random-4 / INT4 | | | |

![quality](figures/quality_vs_precision.png)

### Inference efficiency (real kernels)

| Precision | Model size | Peak VRAM | First token | Tokens/sec |
|---|---|---|---|---|
| FP16 | | | | |
| INT8 | | | | |
| NF4 | | | | |

![efficiency](figures/efficiency_vs_precision.png)

### Layer-wise INT4 sensitivity

![sensitivity](figures/layer_sensitivity.png)

Most sensitive layers: `...`

### What predicts sensitivity?

![stats](figures/stat_correlations.png)

<!-- One or two sentences: which statistic correlated, how strongly, and whether that matched expectation. A weak correlation is a result. -->

### Mixed precision

<!-- Did keeping the top-k layers at INT8 recover quality vs uniform INT4? How did it compare to the random-k control at the same bit budget? -->

## Findings

1.
2.
3.

## Repo layout

```
configs/      experiment config (one YAML per model)
src/          library code: data, model, quant, evaluate, benchmark, stats, utils
scripts/      numbered pipeline steps, each writes CSVs to results/
results/      CSVs (committed), generations/ (gitignored)
figures/      PNGs produced by 08_plots.py
tests/        unit tests for the quantizer
RUNBOOK.md    step-by-step procedure including Kaggle setup
```

## Reproduce

```bash
pip install -r requirements.txt
for s in 01_baseline 02_finetune 03_quantize_eval 04_benchmark 05_sensitivity 06_layer_stats 07_mixed_precision 08_plots; do
  python scripts/$s.py --config configs/qwen0.5b.yaml
done
```

## Not in scope

No CUDA kernels, no QAT, no GPTQ/AWQ reimplementation, no 7B+ models, no serving stack. The contribution is the controlled experimental analysis and the sensitivity-guided mixed-precision layout.
