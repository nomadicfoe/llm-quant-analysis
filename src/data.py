"""Dataset preparation.

Training / held-out loss: Python subset of CodeAlpaca-20k.
Task quality:             MBPP test split, executed against its unit tests.
The two never overlap, so there is no train/eval leakage.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str, inp: str = "") -> str:
    if inp and inp.strip():
        instruction = f"{instruction.strip()}\n\nInput:\n{inp.strip()}"
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def _looks_like_python(ex: dict) -> bool:
    text = (ex["instruction"] + " " + ex.get("input", "")).lower()
    out = ex["output"]
    if "python" in text:
        return True
    return ("def " in out or "import " in out) and "{" not in out and ";" not in out


def load_codealpaca(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Returns (train, heldout) as lists of {"prompt", "response"}.

    The sampled split is cached to results/data_split.json so every script
    sees exactly the same examples.
    """
    cache = Path(cfg["paths"]["results_dir"]) / "data_split.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["train"], d["heldout"]

    ds = load_dataset(cfg["data"]["train_dataset"], split="train")
    rows = [ex for ex in ds if _looks_like_python(ex)]
    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    n_train, n_held = cfg["data"]["n_train"], cfg["data"]["n_heldout"]
    if len(rows) < n_train + n_held:
        raise ValueError(f"Only {len(rows)} python examples after filtering")

    def fmt(ex):
        return {"prompt": format_prompt(ex["instruction"], ex.get("input", "")),
                "response": ex["output"].strip()}

    train = [fmt(ex) for ex in rows[:n_train]]
    heldout = [fmt(ex) for ex in rows[n_train:n_train + n_held]]
    cache.write_text(json.dumps({"train": train, "heldout": heldout}))
    return train, heldout


def load_mbpp(cfg: dict) -> list[dict]:
    """Returns MBPP problems as {"task_id", "prompt", "tests", "setup"}."""
    ds = load_dataset(cfg["data"]["mbpp_dataset"], cfg["data"]["mbpp_config"],
                      split=cfg["data"]["mbpp_split"])
    problems = []
    for ex in ds:
        tests = "\n".join(ex["test_list"])
        instruction = f"{ex['text'].strip()}\nYour code should pass these tests:\n{tests}"
        problems.append({
            "task_id": ex["task_id"],
            "prompt": format_prompt(instruction),
            "tests": ex["test_list"],
            "setup": ex.get("test_setup_code", "") or "",
        })
    return problems[: cfg["data"]["n_mbpp"]]
