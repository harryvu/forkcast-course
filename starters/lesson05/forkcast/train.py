"""Lesson 3. The training toolkit: one epoch, evaluation, fit, save, load (reference solution)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def train_one_epoch(model: nn.Module, X: Tensor, y: Tensor, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    """One pass over the data. Returns the loss as a Python float."""
    model.train()
    optimizer.zero_grad()                 # 1. clean slate (Lesson 2's zero_() for every parameter)
    loss = loss_fn(model(X), y)           #    forward
    loss.backward()                       # 2. slopes into every parameter's .grad
    optimizer.step()                      # 3. the update rule, applied to every parameter
    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, X: Tensor, y: Tensor, loss_fn: nn.Module) -> float:
    """Loss on data the model is not learning from."""
    model.eval()
    return loss_fn(model(X), y).item()


def fit(model: nn.Module, X: Tensor, y: Tensor, epochs: int = 300, lr: float = 0.1,
        optimizer_cls=torch.optim.SGD, loss_fn: nn.Module | None = None) -> list[float]:
    """Build the optimizer, run ``epochs`` epochs, return one loss per epoch."""
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    return [train_one_epoch(model, X, y, optimizer, loss_fn) for _ in range(epochs)]


def save(model: nn.Module, path: str | Path) -> None:
    torch.save(model.state_dict(), path)


def load(model: nn.Module, path: str | Path) -> nn.Module:
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


# ---- Lesson 5: validation, early stopping, regularization ------------------

def fit_with_validation(model: nn.Module, X_train: Tensor, y_train: Tensor, X_val: Tensor, y_val: Tensor,
                        epochs: int = 300, lr: float = 0.01, optimizer_cls=torch.optim.Adam,
                        weight_decay: float = 0.0, patience: int | None = None, loss_fn: nn.Module | None = None) -> dict:
    """Train while watching the validation loss.

    Each epoch: one ``train_one_epoch`` on the training data, then ``evaluate``
    on the validation data. Keep a copy of the best state_dict seen so far
    (``copy.deepcopy(model.state_dict())`` when the validation loss improves).
    If ``patience`` is set, stop when the validation loss has not improved for
    ``patience`` epochs in a row. Before returning, load the best weights back
    into the model. Pass ``weight_decay`` to the optimizer.

    Returns {"train": [...], "val": [...], "best_epoch": int, "best_val": float}.
    """
    raise NotImplementedError("Lesson 5, step 3: the loop above; import copy for deepcopy")
