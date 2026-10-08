"""Lesson 6. Real deliveries: taxi trips as a stand-in for Forkcast ETAs.

``clean_trips`` and ``load_trips`` are provided and apply the same rules to the
bundled sample and to a real month. ``TripsDataset`` and ``make_loaders`` are
the exercises.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

CONTINUOUS = ["trip_distance", "pickup_hour", "weekday", "passenger_count"]
CATEGORICAL = ["pu_zone", "do_zone"]
ZONE_VOCAB = 266          # TLC zone ids run 1..265; index 0 is reserved for "unknown"
TARGET = "minutes"
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "taxi_sample.csv"


def clean_trips(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Turn TLC columns into model columns and drop impossible rows.

    Rules, stated once so they can be argued with: a trip must move (distance 0 to 50 miles),
    take between 1 and 120 minutes, and have a known or explicitly unknown zone.
    """
    frame = raw.copy()
    pickup = pd.to_datetime(frame["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(frame["tpep_dropoff_datetime"])
    frame[TARGET] = (dropoff - pickup).dt.total_seconds() / 60
    frame["pickup_hour"] = pickup.dt.hour.astype("float32")
    frame["weekday"] = pickup.dt.dayofweek.astype("float32")
    frame["passenger_count"] = frame["passenger_count"].fillna(1).clip(1, 6).astype("float32")
    frame["trip_distance"] = frame["trip_distance"].astype("float32")
    for src, dst in (("PULocationID", "pu_zone"), ("DOLocationID", "do_zone")):
        zone = frame[src].fillna(0).astype(int)
        frame[dst] = zone.where((zone >= 1) & (zone <= 265), 0)          # unknown -> 0
    before = len(frame)
    keep = (frame["trip_distance"] > 0) & (frame["trip_distance"] <= 50) & (frame[TARGET] >= 1) & (frame[TARGET] <= 120)
    frame = frame.loc[keep, CONTINUOUS + CATEGORICAL + [TARGET]].reset_index(drop=True)
    if verbose:
        print(f"clean_trips: kept {len(frame):,} of {before:,} rows ({before - len(frame):,} removed)")
    return frame


def load_trips(path: str | Path = SAMPLE, verbose: bool = True) -> pd.DataFrame:
    """Read a CSV or parquet of TLC trips and clean it."""
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return clean_trips(raw, verbose=verbose)


def split_frame(frame: pd.DataFrame, fractions=(0.7, 0.15, 0.15), seed: int = 0):
    """Shuffle rows and cut into three frames (Lesson 5's split, for DataFrames)."""
    perm = torch.randperm(len(frame), generator=torch.Generator().manual_seed(seed)).numpy()
    n_train, n_val = round(len(frame) * fractions[0]), round(len(frame) * fractions[1])
    parts = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    return tuple(frame.iloc[idx].reset_index(drop=True) for idx in parts)


class TripsDataset(Dataset):
    """One trip per item: (continuous features (4,), zone ids (2,) as int64, minutes ()).

    ``normalizer`` is a fitted Normalizer for the continuous columns (fit it on the
    training frame only). Store everything as tensors in ``__init__`` so ``__getitem__``
    is a plain index.
    """

    def __init__(self, frame: pd.DataFrame, normalizer=None):
        raise NotImplementedError("Lesson 6, step 3: cont = torch.tensor(frame[CONTINUOUS].values, dtype=torch.float32), normalized; cat = ... .long(); target = ...")

    def __len__(self) -> int:
        raise NotImplementedError("Lesson 6, step 3: how many trips?")

    def __getitem__(self, i: int):
        raise NotImplementedError("Lesson 6, step 3: return self.cont[i], self.cat[i], self.target[i]")


def make_loaders(train_frame: pd.DataFrame, val_frame: pd.DataFrame, normalizer, batch_size: int = 256) -> tuple[DataLoader, DataLoader]:
    """Training loader: shuffled, drop_last=True. Validation loader: in order, every row."""
    raise NotImplementedError("Lesson 6, step 4: two TripsDatasets, two DataLoaders")
