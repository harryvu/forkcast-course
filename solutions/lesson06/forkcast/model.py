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
    """A multilayer perceptron: Linear -> activation [-> Dropout] -> ... -> Linear(…, 1).

    ``dropout`` (Lesson 5) is the fraction of hidden units zeroed on each training step; 0 adds no Dropout layers.
    """

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (32, 32), activation=nn.ReLU, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers.append(nn.Linear(width, h))        # (N, width) -> (N, h)
            layers.append(activation())               # bends; without it two Linears are one Linear
            if dropout > 0:
                layers.append(nn.Dropout(dropout))    # training only: randomly zero a fraction of the units
            width = h
        layers.append(nn.Linear(width, 1))            # the output layer has no activation
        self.net = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        """(N, d) -> (N,)"""
        return self.net(X).squeeze(-1)


# ---- Lesson 6: continuous features plus embedded zones ------------------------

class TabularNet(nn.Module):
    """Continuous features and two zone ids -> minutes.

    Each zone id (0..zone_vocab-1) is looked up in an nn.Embedding table of shape
    (zone_vocab, emb_dim). The two embedded zones are concatenated with the
    continuous features and fed to an MLP: input width n_continuous + 2 * emb_dim.
    """

    def __init__(self, n_continuous: int, zone_vocab: int = 266, emb_dim: int = 8, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.pu = nn.Embedding(zone_vocab, emb_dim)      # a learned table: one row of emb_dim numbers per zone
        self.do = nn.Embedding(zone_vocab, emb_dim)
        self.mlp = MLP(n_continuous + 2 * emb_dim, hidden)

    def forward(self, x_cont: Tensor, x_cat: Tensor) -> Tensor:
        """(N, n_continuous) float, (N, 2) int64 -> (N,)"""
        features = torch.cat([x_cont, self.pu(x_cat[:, 0]), self.do(x_cat[:, 1])], dim=1)   # (N, 4 + 8 + 8)
        return self.mlp(features)
