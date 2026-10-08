"""Lesson 3. Feature normalization with statistics learned from training data only."""

from __future__ import annotations

import torch
from torch import Tensor


class Normalizer:
    """Give every feature column mean 0 and standard deviation 1.

    ``fit`` learns the per-column mean and std from one tensor (the training
    data). ``transform`` applies those stored statistics to any tensor, so
    validation and test data are scaled the same way as the training data.
    """

    def __init__(self):
        self.mean: Tensor | None = None
        self.std: Tensor | None = None

    def fit(self, X: Tensor) -> "Normalizer":
        """Store the column means and population stds (``correction=0``) of X, shape (d,) each.

        Clamp the std with ``.clamp_min(1e-6)`` so a constant column cannot cause a division by zero.
        Return ``self`` so ``Normalizer().fit(X)`` works.
        """
        raise NotImplementedError("Lesson 3, step 4: self.mean = ..., self.std = ...; return self")

    def transform(self, X: Tensor) -> Tensor:
        """(N, d) -> (N, d) using the stored statistics. Must not recompute them."""
        raise NotImplementedError("Lesson 3, step 4: (X - self.mean) / self.std")

    def fit_transform(self, X: Tensor) -> Tensor:
        return self.fit(X).transform(X)
