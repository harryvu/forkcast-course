"""Lesson 17 (optional). Experiment hygiene: runs you can reproduce, compare and trust.

``git_commit`` and ``load_log`` are provided; the rest are exercises.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed every source of randomness a training run uses, and return a generator for DataLoaders.

    Python's ``random``, NumPy, torch (CPU and every CUDA device). With ``deterministic`` also call
    ``torch.use_deterministic_algorithms(deterministic, warn_only=True)`` and set
    ``torch.backends.cudnn.benchmark = not deterministic``. Return ``torch.Generator().manual_seed(seed)``,
    which a DataLoader takes as ``generator=`` so its shuffle order is reproducible too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                               # seeds the CPU generator and every CUDA device
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic   # benchmark picks kernels by timing: fast, but a source of run-to-run drift
    return torch.Generator().manual_seed(seed)


@dataclass(frozen=True)
class RunConfig:
    """Everything that decides a result, in one object. Change a field and the fingerprint changes."""
    model: str = "mlp"                      # "mlp" or "linear"
    hidden: tuple = (32, 32)
    lr: float = 0.01
    epochs: int = 150
    weight_decay: float = 0.0
    dropout: float = 0.0
    n_orders: int = 2000
    data_seed: int = 0
    seed: int = 0

    def to_json(self, path: str | Path) -> None:
        """Write the config as JSON (tuples become lists)."""
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        """Read a config back; turn the hidden list into a tuple."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["hidden"] = tuple(data["hidden"])
        return cls(**data)

    def fingerprint(self) -> str:
        """First 8 hex characters of the SHA-256 of ``json.dumps(asdict(self), sort_keys=True)``: the run's identity."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]


def git_commit() -> str | None:
    """The current commit, so a result can be traced to the code that produced it; None outside a repository."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except Exception:
        return None


def run_experiment(config: RunConfig, log_path: str | Path | None = None) -> dict:
    """Train one model from a config, reproducibly, and append the result to a JSON-lines log.

    seed_everything(config.seed); make_orders(config.n_orders, seed=config.data_seed, nonlinear=True);
    train_val_test_split(seed=config.data_seed); Normalizer fitted on the training part; the model is
    LinearRegressor(4) or MLP(4, hidden, dropout=...); fit_with_validation(epochs, lr, weight_decay).
    Returns {"fingerprint", "best_val", "best_epoch", "seconds", "commit", "timestamp", **asdict(config)}
    and, when log_path is given, appends it as one JSON line.
    """
    from forkcast import train
    from forkcast.data import make_orders
    from forkcast.model import MLP, LinearRegressor
    from forkcast.preprocess import Normalizer
    from forkcast.split import train_val_test_split

    seed_everything(config.seed)
    start = time.perf_counter()
    X, y = make_orders(config.n_orders, seed=config.data_seed, nonlinear=True)
    (X_tr, y_tr), (X_val, y_val), _ = train_val_test_split(X, y, seed=config.data_seed)
    norm = Normalizer().fit(X_tr)
    model = LinearRegressor(4) if config.model == "linear" else MLP(4, tuple(config.hidden), dropout=config.dropout)
    history = train.fit_with_validation(model, norm.transform(X_tr), y_tr, norm.transform(X_val), y_val,
                                        epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay)
    result = {"fingerprint": config.fingerprint(), "best_val": history["best_val"], "best_epoch": history["best_epoch"],
              "seconds": time.perf_counter() - start, "commit": git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              **asdict(config)}
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")               # one line per run: append-only, diff-friendly, greppable
    return result


def load_log(path: str | Path) -> pd.DataFrame:
    """The experiment log as a DataFrame, one row per run."""
    return pd.read_json(path, lines=True)


def compare(log: pd.DataFrame) -> pd.DataFrame:
    """Group the log by everything but the seed and the per-run fields; report count, mean and std of best_val.

    Group keys: model, hidden (as a string), lr, epochs, weight_decay, dropout, n_orders, data_seed.
    Columns out: those keys plus runs, mean_val, std_val, sorted by mean_val.
    """
    keys = ["model", "hidden", "lr", "epochs", "weight_decay", "dropout", "n_orders", "data_seed"]
    frame = log.assign(hidden=log["hidden"].astype(str))
    out = frame.groupby(keys)["best_val"].agg(runs="count", mean_val="mean", std_val="std").reset_index()
    return out.sort_values("mean_val").reset_index(drop=True)


def repeat_with_seeds(config: RunConfig, seeds, log_path: str | Path | None = None) -> dict:
    """Run the same config under several seeds. Returns {"values": [...], "mean": float, "std": float} (sample std, ddof=1)."""
    from dataclasses import replace
    values = [run_experiment(replace(config, seed=int(s)), log_path)["best_val"] for s in seeds]
    t = torch.tensor(values, dtype=torch.float64)
    return {"values": values, "mean": t.mean().item(), "std": t.std().item() if len(values) > 1 else 0.0}
