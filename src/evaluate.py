"""Quality evaluation.

mbpp_pass_at_1: greedy-generate a solution per problem, run it against the MBPP
                unit tests in a subprocess with a timeout. Fraction that pass.
heldout_loss:   mean cross-entropy over response tokens of held-out CodeAlpaca
                examples. Cheap, deterministic, used for the layer sweep.
"""
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    text = text.split("### Instruction")[0]  # model continued into a new prompt
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.replace("```", "").strip()


def run_tests(code: str, problem: dict, timeout: int) -> bool:
    src = f"{problem['setup']}\n\n{code}\n\n" + "\n".join(problem["tests"]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


class StopOnMarker(StoppingCriteria):
    """Stop once every sequence in the batch has emitted EOS or started a new '###' block."""

    def __init__(self, tok, prompt_len: int, marker: str = "###"):
        self.tok, self.prompt_len, self.marker = tok, prompt_len, marker

    def __call__(self, input_ids, scores, **kwargs):
        new = input_ids[:, self.prompt_len:]
        done = []
        for row in new:
            if (row == self.tok.eos_token_id).any():
                done.append(True)
                continue
            done.append(self.marker in self.tok.decode(row[-8:], skip_special_tokens=True))
        return torch.tensor(done, device=input_ids.device)


@torch.no_grad()
def generate(model, tok, prompts: list[str], cfg: dict) -> list[str]:
    outs = []
    bs = cfg["eval"]["batch_size"]
    for i in tqdm(range(0, len(prompts), bs), desc="generate", leave=False):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        plen = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=cfg["eval"]["max_new_tokens"],
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnMarker(tok, plen)]),
        )
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outs


def mbpp_pass_at_1(model, tok, problems: list[dict], cfg: dict,
                   save_to: str | None = None) -> float:
    raw = generate(model, tok, [p["prompt"] for p in problems], cfg)
    codes = [extract_code(r) for r in raw]
    with ThreadPoolExecutor(cfg["eval"]["exec_workers"]) as ex:
        results = list(ex.map(
            lambda c_p: run_tests(c_p[0], c_p[1], cfg["eval"]["exec_timeout"]),
            zip(codes, problems)))
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, "w") as f:
            for p, r, c, ok in zip(problems, raw, codes, results):
                f.write(json.dumps({"task_id": p["task_id"], "raw": r,
                                    "code": c, "passed": ok}) + "\n")
    return sum(results) / len(results)


@torch.no_grad()
def heldout_loss(model, tok, examples: list[dict], cfg: dict) -> float:
    """Mean token-level NLL over response tokens only."""
    tok.padding_side = "right"
    total_loss, total_tok = 0.0, 0
    bs = cfg["eval"]["loss_batch_size"]
    max_len = cfg["data"]["max_seq_len"]
    for i in tqdm(range(0, len(examples), bs), desc="loss", leave=False):
        batch = examples[i:i + bs]
        ids_list, lab_list = [], []
        for ex in batch:
            p = tok(ex["prompt"], add_special_tokens=False).input_ids
            r = tok(ex["response"] + tok.eos_token, add_special_tokens=False).input_ids
            ids = (p + r)[:max_len]
            lab = ([-100] * len(p) + r)[:max_len]
            ids_list.append(ids)
            lab_list.append(lab)
        L = max(len(x) for x in ids_list)
        input_ids = torch.full((len(batch), L), tok.pad_token_id)
        labels = torch.full((len(batch), L), -100)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (ids, lab) in enumerate(zip(ids_list, lab_list)):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            labels[j, :len(lab)] = torch.tensor(lab)
            attn[j, :len(ids)] = 1
        input_ids, labels, attn = (t.to(model.device) for t in (input_ids, labels, attn))
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float()
        tgt = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=-100, reduction="sum")
        total_loss += loss.item()
        total_tok += (tgt != -100).sum().item()
    tok.padding_side = "left"
    return total_loss / total_tok
