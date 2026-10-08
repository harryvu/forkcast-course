"""Lesson 5. Splitting data into training, validation and test sets (reference solution)."""

from __future__ import annotations

import torch
from torch import Tensor

Split = tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]


def train_val_test_split(X: Tensor, y: Tensor, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0) -> Split:
    """Shuffle the rows (seeded) and cut them into three parts."""
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))   # a random order of the row indices
    n_train = round(n * fractions[0])
    n_val = round(n * fractions[1])
    idx_train, idx_val, idx_test = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    return (X[idx_train], y[idx_train]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])
