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

# ---- Lesson 4: the first neural network -------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in ``model`` (sum of numel over parameters that require grad)."""
    raise NotImplementedError("Lesson 4, step 4: sum(p.numel() for p in model.parameters() if p.requires_grad)")


class MLP(nn.Module):
    """A multilayer perceptron: Linear -> activation -> ... -> Linear(…, 1).

    ``hidden`` lists the width of each hidden layer, so MLP(4, (32, 32)) is
    4 -> 32 -> 32 -> 1 with an activation after each hidden layer and none
    after the last. ``activation`` is a class such as nn.ReLU; MLP(4, ())
    has no hidden layers and is exactly LinearRegressor.
    """

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (32, 32), activation=nn.ReLU):
        super().__init__()
        raise NotImplementedError("Lesson 4, step 4: build a list of layers, wrap it in nn.Sequential as self.net")

    def forward(self, X: Tensor) -> Tensor:
        """(N, d) -> (N,)"""
        raise NotImplementedError("Lesson 4, step 4: return self.net(X).squeeze(-1)")
