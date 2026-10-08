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
    """Train while watching the validation loss; keep the best weights; optionally stop early."""
    import copy

    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": [], "best_epoch": 0, "best_val": float("inf")}
    best_state = copy.deepcopy(model.state_dict())
    since_best = 0
    for epoch in range(epochs):
        history["train"].append(train_one_epoch(model, X_train, y_train, optimizer, loss_fn))
        val_loss = evaluate(model, X_val, y_val, loss_fn)
        history["val"].append(val_loss)
        if val_loss < history["best_val"]:
            history["best_val"], history["best_epoch"] = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())     # a snapshot, not a reference
            since_best = 0
        else:
            since_best += 1
            if patience is not None and since_best >= patience:
                break                                           # early stopping
    model.load_state_dict(best_state)                           # hand back the best model, not the last one
    return history


# ---- Lesson 6: the toolkit learns to take a DataLoader ------------------------

def _split_batch(batch, device):
    """A batch is a tuple (input_1, ..., input_k, target). Move everything to ``device``."""
    *inputs, target = batch
    return [x.to(device) for x in inputs], target.to(device)


def train_one_epoch_loader(model: nn.Module, loader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: str | torch.device = "cpu") -> float:
    """One pass over every batch in ``loader``. Returns the mean training loss, weighted by batch size.

    For each batch: split it, zero_grad, forward with ``model(*inputs)``, loss, backward, step.
    """
    raise NotImplementedError("Lesson 6, step 6: loop over loader; accumulate loss.item() * len(target); divide by the number of rows seen")


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, loss_fn: nn.Module, device: str | torch.device = "cpu") -> float:
    """Mean loss over every batch in ``loader``, weighted by batch size, in eval mode."""
    raise NotImplementedError("Lesson 6, step 6: model.eval(); the same loop without the optimizer")


def fit_loader(model: nn.Module, train_loader, val_loader, epochs: int = 20, lr: float = 1e-3, optimizer_cls=torch.optim.Adam,
               weight_decay: float = 0.0, patience: int | None = None, loss_fn: nn.Module | None = None, device: str | torch.device = "cpu") -> dict:
    """Lesson 5's fit_with_validation, with loaders instead of tensors. Same history, same early stopping, same best-weights rule."""
    raise NotImplementedError("Lesson 6, step 6: copy fit_with_validation and swap in the loader functions; move the model to device first")
