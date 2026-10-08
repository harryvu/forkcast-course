"""Lesson 3. Feature normalization with statistics learned from training data only (reference solution)."""

from __future__ import annotations

import torch
from torch import Tensor


class Normalizer:
    """Give every feature column mean 0 and standard deviation 1."""

    def __init__(self):
        self.mean: Tensor | None = None
        self.std: Tensor | None = None

    def fit(self, X: Tensor) -> "Normalizer":
        self.mean = X.mean(dim=0)                               # (d,)
        self.std = X.std(dim=0, correction=0).clamp_min(1e-6)   # (d,)
        return self

    def transform(self, X: Tensor) -> Tensor:
        return (X - self.mean) / self.std                       # (N, d) - (d,) broadcasts over rows

    def fit_transform(self, X: Tensor) -> Tensor:
        return self.fit(X).transform(X)
