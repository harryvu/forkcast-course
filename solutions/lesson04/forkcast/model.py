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

# ---- Lesson 4: the first neural network -------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    """A multilayer perceptron: Linear -> activation -> ... -> Linear(…, 1)."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (32, 32), activation=nn.ReLU):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers.append(nn.Linear(width, h))        # (N, width) -> (N, h)
            layers.append(activation())               # bends; without it two Linears are one Linear
            width = h
        layers.append(nn.Linear(width, 1))            # the output layer has no activation
        self.net = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        """(N, d) -> (N,)"""
        return self.net(X).squeeze(-1)
