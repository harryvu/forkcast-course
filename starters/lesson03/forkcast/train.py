"""Lesson 3. The training toolkit: one epoch, evaluation, fit, save, load.

These functions are reused by every later lesson. Right now the whole dataset
fits in one batch; Lesson 6 replaces X, y with a DataLoader.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def train_one_epoch(model: nn.Module, X: Tensor, y: Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    """One pass over the data. Returns the loss as a Python float.

    The three lines every PyTorch training step has, in this order:
    zero the gradients, backward, step. Put the model in training mode first.
    """
    raise NotImplementedError("Lesson 3, step 5: model.train(); zero_grad; forward; loss; backward; step; return loss.item()")


@torch.no_grad()
def evaluate(model: nn.Module, X: Tensor, y: Tensor, loss_fn: nn.Module) -> float:
    """Loss on data the model is not learning from. No gradients are needed or wanted."""
    raise NotImplementedError("Lesson 3, step 5: model.eval(); return loss_fn(model(X), y).item()")


def fit(model: nn.Module, X: Tensor, y: Tensor, epochs: int = 300, lr: float = 0.1,
        optimizer_cls=torch.optim.SGD, loss_fn: nn.Module | None = None) -> list[float]:
    """Build the optimizer over ``model.parameters()``, run ``epochs`` epochs, return one loss per epoch."""
    raise NotImplementedError("Lesson 3, step 5: optimizer = optimizer_cls(model.parameters(), lr=lr); loop train_one_epoch")


def save(model: nn.Module, path: str | Path) -> None:
    """Write the model's state_dict to ``path``."""
    raise NotImplementedError("Lesson 3, step 8: torch.save(model.state_dict(), path)")


def load(model: nn.Module, path: str | Path) -> nn.Module:
    """Load a state_dict from ``path`` into ``model`` (built with the same n_features) and return it."""
    raise NotImplementedError("Lesson 3, step 8: model.load_state_dict(torch.load(path, weights_only=True)); return model")
