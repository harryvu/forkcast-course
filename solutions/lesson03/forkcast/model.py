"""Lesson 3. The delivery-time model as an nn.Module (reference solution)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearRegressor(nn.Module):
    """minutes = X @ w + b, with the knobs owned by an nn.Linear layer."""

    def __init__(self, n_features: int):
        super().__init__()                        # registers parameters, enables .to(), state_dict, ...
        self.linear = nn.Linear(n_features, 1)    # weight (1, d), bias (1,)

    def forward(self, X: Tensor) -> Tensor:
        """(N, d) -> (N,)"""
        return self.linear(X).squeeze(-1)         # (N, 1) -> (N,), so it lines up with y
