# RUNBOOK: what to do, in order

Every step is one script. Each writes CSVs into `results/`. Scripts are resumable: rerun after a crash and finished rows are skipped.

Total GPU time on a Kaggle T4: roughly 10 to 12 hours across 3 or 4 sessions.

---

## 0. One-time setup (30 min, no GPU)

### 0.1 GitHub
1. Create an empty public repo on GitHub, e.g. `llm-quant-analysis`.
2. Unzip this project, then:
   ```bash
   cd llm-quant-analysis
   git init && git add . && git commit -m "initial pipeline"
   git branch -M main
   git remote add origin https://github.com/nomadicfoe/llm-quant-analysis.git
   git push -u origin main
   ```

### 0.2 Hugging Face
1. Create an account at huggingface.co if you don't have one.
2. Settings > Access Tokens > create a read token. Save it.
   (Qwen2.5-0.5B is not gated, but a token avoids rate limits and is needed for some datasets.)

### 0.3 Kaggle
1. kaggle.com > Settings > Phone verification. Without this you get no GPU.
2. Create a new Notebook.
3. Right panel: Accelerator = **GPU T4 x2**, Internet = **On**, Persistence = **Files only**.
4. Add-ons > Secrets > add `HF_TOKEN` = your Hugging Face token.
5. Also add a secret `GH_TOKEN` = a GitHub personal access token with `repo` scope. Used to push results back at the end of each session.

### 0.4 Local sanity check (optional, CPU, 1 min)
```bash
pip install -r requirements.txt
pytest -q          # 5 quant tests should pass
```

---

## 1. Kaggle session template

Paste this as the first cell of every session:

```python
from kaggle_secrets import UserSecretsClient
import os, subprocess
s = UserSecretsClient()
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")
GH = s.get_secret("GH_TOKEN")
REPO = f"https://{GH}@github.com/nomadicfoe/llm-quant-analysis.git"

!git clone {REPO} /kaggle/working/proj 2>/dev/null || (cd /kaggle/working/proj && git pull)
%cd /kaggle/working/proj
!pip install -q -r requirements.txt
!git config user.email "you@example.com" && git config user.name "Sumanth Paila"
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Paste this as the LAST cell of every session and run it before the session ends:

```python
%cd /kaggle/working/proj
!git add results figures && git commit -m "results: session $(date +%F)" && git push
```

The working disk is wiped when the session ends. CSVs live in git, checkpoints do not (see step 2.4).

Kaggle sessions cap at 12 hours. The sweep in step 5 is the only step that can approach that.

---

## 2. Session 1: baseline + fine-tune (about 1.5 GPU hours)

### 2.1 Smoke test the whole pipeline first (15 min)
```bash
!python scripts/01_baseline.py --smoke
!python scripts/02_finetune.py --smoke
!python scripts/03_quantize_eval.py --smoke
!python scripts/08_plots.py
```
Check `results/quality.csv` has 4 rows and `figures/` has PNGs. If anything errors, fix it now on the small run.

Then delete the smoke outputs so they don't pollute real results:
```bash
!rm -rf results/*.csv results/generations results/data_split.json checkpoints figures/*.png
```

### 2.2 Baseline (15 min)
```bash
!python scripts/01_baseline.py
```
Before: nothing.
Output: `results/quality.csv` row `base_fp16`, `results/generations/base_fp16.jsonl`, `results/data_split.json`.
Expect: pass@1 somewhere in 15 to 30% for a 0.5B base model. Open a few lines of the jsonl and confirm the generated code looks like code. If pass@1 is 0, something is wrong with prompt formatting or code extraction, do not proceed.

### 2.3 Fine-tune (45 min train + 15 min eval)
```bash
!python scripts/02_finetune.py
```
Before: `01_baseline.py` finished.
Output: `checkpoints/lora/`, `checkpoints/merged_fp16/`, `quality.csv` row `finetuned_fp16`.
Expect: training loss falling from ~1.2 to ~0.6. pass@1 higher than base. If it is not, try `epochs: 3` or `lr: 3.0e-4` in the config, delete the checkpoint dirs, rerun. Do not move on until fine-tuned beats base, the whole story depends on it.

### 2.4 Save the merged checkpoint (5 min)
Checkpoints are gitignored (1 GB). Save them as a Kaggle Dataset so later sessions skip retraining:
```python
!mkdir -p /kaggle/working/ckpt && cp -r checkpoints/merged_fp16 /kaggle/working/ckpt/
```
Then in the Kaggle UI: Output tab > `ckpt` folder > "Create Dataset" > name it `llm-quant-ckpt`.

In every later session, add that dataset as input and restore it:
```bash
!mkdir -p checkpoints && cp -r /kaggle/input/llm-quant-ckpt/merged_fp16 checkpoints/
```

### 2.5 Uniform quantization (30 min)
```bash
!python scripts/03_quantize_eval.py
```
Output: `quality.csv` rows `finetuned_int8`, `finetuned_int4`.
Expect: INT8 within ~1 point of FP16, INT4 a visible drop. If INT4 shows zero drop, change `int4_group_size` to 256 or note it as a finding.

Run the push cell. You now have a usable resume line.

---

## 3. Session 2: efficiency benchmark (30 min)

Restore the checkpoint (step 2.4), then:
```bash
!python scripts/04_benchmark.py
```
Output: `results/efficiency.csv`, one row per precision.
Expect: memory drops FP16 > INT8 > INT4. On a T4, INT8 tokens/sec is often *slower* than FP16 because bitsandbytes LLM.int8 kernels are not fast on that card. That is a real, well-known result. Report it, don't hide it.

If bitsandbytes fails to import, run `!pip install -U bitsandbytes` and retry.

Push.

---

## 4. Session 3: layer sensitivity sweep (4 to 6 hours)

Restore checkpoint, then:
```bash
!python scripts/05_sensitivity.py
```
Pass A runs held-out loss for all 24 layers (~2 min each, ~50 min).
Pass B runs MBPP pass@1 on the 6 most loss-sensitive layers (~15 min each, ~1.5 h).
Set `sensitivity.with_pass1: true` in the config if you want pass@1 on all 24 layers (adds ~4.5 h; do it in a separate session if you want the full bar chart).

Output: `results/layer_sensitivity.csv` (+ the two partial CSVs it is built from).
Expect: a handful of layers with clearly higher loss delta. Typically first and last few layers, but that is what you are finding out.

Then, no GPU needed for long:
```bash
!python scripts/06_layer_stats.py
```
Output: `results/layer_stats.csv`, `results/stat_correlations.csv`.
Read the log: it prints Pearson and Spearman correlation for every statistic against sensitivity. Weak correlations are a valid finding.

Push.

---

## 5. Session 4: mixed precision + figures (1.5 hours)

Restore checkpoint, then:
```bash
!python scripts/07_mixed_precision.py
!python scripts/08_plots.py
```
Output: `quality.csv` rows `mixed_top4`, `mixed_random4`, `mixed_top8`, `mixed_random8`; all figures.
Expect: `mixed_topK` above `finetuned_int4` and above `mixed_randomK` at nearly the same average bits. The random control is what proves the sensitivity ranking is doing work.

Push. Then write the README results section (see README.md template) with your real numbers.

---

## 6. Optional: rerun on the 1.5B model

```bash
cp configs/qwen0.5b.yaml configs/qwen1.5b.yaml
# edit: model.name: Qwen/Qwen2.5-1.5B, n_layers: 28
# edit paths: results_dir: results_1.5b, figures_dir: figures_1.5b, adapter_dir/merged_dir similarly
python scripts/01_baseline.py --config configs/qwen1.5b.yaml   # and so on
```
About 3x the GPU time. Only do this after 0.5B is complete and written up.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM in 02_finetune | `batch_size: 4`, `grad_accum: 4` |
| CUDA OOM in eval | `eval.batch_size: 8` |
| pass@1 = 0 for base | inspect `results/generations/base_fp16.jsonl`, check `extract_code` output |
| Fine-tuned not better than base | `epochs: 3`, or check train loss actually decreased |
| `bitsandbytes` import error | `pip install -U bitsandbytes`, restart kernel |
| Session died mid-sweep | just rerun `05_sensitivity.py`, it resumes |
| Kaggle "no GPU quota" | 30 h/week resets on Saturday; or switch to Colab Pro, same commands |
