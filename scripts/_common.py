import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
"""Shared argument parsing for all scripts. Import this first."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, set_seed, get_logger  # noqa: E402


def setup(name: str):
    ap = argparse.ArgumentParser(description=name)
    ap.add_argument("--config", default="configs/qwen0.5b.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to check the pipeline (100 MBPP, 64 heldout, 200 train)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["data"]["n_mbpp"] = 100
        cfg["data"]["n_heldout"] = min(cfg["data"]["n_heldout"], 64)
        cfg["train"]["epochs"] = 1
        cfg["_smoke"] = True
    set_seed(cfg["seed"])
    return cfg, get_logger(name)
