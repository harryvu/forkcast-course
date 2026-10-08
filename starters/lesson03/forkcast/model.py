"""Lesson 3. The delivery-time model as an nn.Module.

Replace every ``raise NotImplementedError`` with one to three lines of code.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearRegressor(nn.Module):
    """minutes = X @ w + b, with the knobs owned by an nn.Linear layer."""

    def __init__(self, n_features: int):
        super().__init__()
        raise NotImplementedError("Lesson 3, step 3: self.linear = nn.Linear(n_features, 1)")

    def forward(self, X: Tensor) -> Tensor:
        """(N, d) -> (N,). nn.Linear returns (N, 1); squeeze the last dimension away."""
        raise NotImplementedError("Lesson 3, step 3: return self.linear(X).squeeze(-1)")
