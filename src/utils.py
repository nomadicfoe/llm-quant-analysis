"""Shared helpers: config loading, seeding, CSV writing, logging."""
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        abs_path = ROOT / rel
        abs_path.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(abs_path)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def append_csv(path: str, row: dict) -> None:
    """Append one row. Writes header if the file is new. Every experiment writes through this."""
    path = Path(path)
    existing = read_csv(path)
    if not existing and path.exists() and path.stat().st_size == 0:
        path.unlink()
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return
    header = list(existing[0].keys()) if existing else _header(path)
    if set(row) - set(header):
        # new column appeared: rewrite with the union header
        header = header + [k for k in row if k not in header]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in existing:
                w.writerow({k: r.get(k, "") for k in header})
            w.writerow({k: row.get(k, "") for k in header})
        return
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow({k: row.get(k, "") for k in header})


def _header(path: Path) -> list[str]:
    with open(path, newline="") as f:
        return next(csv.reader(f))


def read_csv(path: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def done_values(path: str, column: str) -> set:
    """Values already present in a CSV column. Used to make sweeps resumable."""
    return {r[column] for r in read_csv(path)}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
