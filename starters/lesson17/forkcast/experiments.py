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
    raise NotImplementedError("Lesson 17, step 2: random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all, determinism flags, return a Generator")


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
        raise NotImplementedError("Lesson 17, step 3: Path(path).write_text(json.dumps(asdict(self), indent=2))")

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        """Read a config back; turn the hidden list into a tuple."""
        raise NotImplementedError("Lesson 17, step 3: data = json.loads(...); data[\"hidden\"] = tuple(data[\"hidden\"]); return cls(**data)")

    def fingerprint(self) -> str:
        """First 8 hex characters of the SHA-256 of ``json.dumps(asdict(self), sort_keys=True)``: the run's identity."""
        raise NotImplementedError("Lesson 17, step 3: hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]")


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
    raise NotImplementedError("Lesson 17, step 4: seed, data, split, normalize, model, fit_with_validation, result dict, append a JSON line")


def load_log(path: str | Path) -> pd.DataFrame:
    """The experiment log as a DataFrame, one row per run."""
    return pd.read_json(path, lines=True)


def compare(log: pd.DataFrame) -> pd.DataFrame:
    """Group the log by everything but the seed and the per-run fields; report count, mean and std of best_val.

    Group keys: model, hidden (as a string), lr, epochs, weight_decay, dropout, n_orders, data_seed.
    Columns out: those keys plus runs, mean_val, std_val, sorted by mean_val.
    """
    raise NotImplementedError("Lesson 17, step 5: log.assign(hidden=...astype(str)).groupby(keys)[\"best_val\"].agg([\"count\", \"mean\", \"std\"])...")


def repeat_with_seeds(config: RunConfig, seeds, log_path: str | Path | None = None) -> dict:
    """Run the same config under several seeds. Returns {"values": [...], "mean": float, "std": float} (sample std, ddof=1)."""
    raise NotImplementedError("Lesson 17, step 5: dataclasses.replace(config, seed=s) for each seed; run_experiment; mean and std")
