"""Lesson 5. Splitting data into training, validation and test sets."""

from __future__ import annotations

import torch
from torch import Tensor

Split = tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]


def train_val_test_split(X: Tensor, y: Tensor, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0) -> Split:
    """Shuffle the rows with ``torch.randperm`` (seeded) and cut them into three parts.

    Returns ((X_train, y_train), (X_val, y_val), (X_test, y_test)). Sizes are
    round(N * fraction) for train and validation; the test set gets the rest, so
    every row lands in exactly one part.
    """
    raise NotImplementedError("Lesson 5, step 2: perm = torch.randperm(N, generator=...); slice perm into three index tensors; return X[idx], y[idx] for each")
