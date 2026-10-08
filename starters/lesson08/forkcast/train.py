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
    model.train()
    total, seen = 0.0, 0
    for batch in loader:
        inputs, target = _split_batch(batch, device)
        optimizer.zero_grad()
        loss = loss_fn(model(*inputs), target)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)          # weight by batch size: the last batch may be smaller
        seen += len(target)
    return total / seen


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, loss_fn: nn.Module, device: str | torch.device = "cpu") -> float:
    """Mean loss over every batch in ``loader``, weighted by batch size, in eval mode."""
    model.eval()
    total, seen = 0.0, 0
    for batch in loader:
        inputs, target = _split_batch(batch, device)
        total += loss_fn(model(*inputs), target).item() * len(target)
        seen += len(target)
    return total / seen


def fit_loader(model: nn.Module, train_loader, val_loader, epochs: int = 20, lr: float = 1e-3, optimizer_cls=torch.optim.Adam,
               weight_decay: float = 0.0, patience: int | None = None, loss_fn: nn.Module | None = None, device: str | torch.device = "cpu") -> dict:
    """Lesson 5's fit_with_validation, with loaders instead of tensors. Same history, same early stopping, same best-weights rule."""
    import copy

    model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": [], "best_epoch": 0, "best_val": float("inf")}
    best_state = copy.deepcopy(model.state_dict())
    since_best = 0
    for epoch in range(epochs):
        history["train"].append(train_one_epoch_loader(model, train_loader, optimizer, loss_fn, device))
        val_loss = evaluate_loader(model, val_loader, loss_fn, device)
        history["val"].append(val_loss)
        if val_loss < history["best_val"]:
            history["best_val"], history["best_epoch"] = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
            if patience is not None and since_best >= patience:
                break
    model.load_state_dict(best_state)
    return history


# ---- Lesson 8: devices, mixed precision, checkpoints, logging: training loop v2 ----

import time


def autocast_dtype(device: str | torch.device) -> torch.dtype:
    """float16 on a GPU (fast tensor cores), bfloat16 on a CPU (same range as float32, no scaler needed)."""
    return torch.float16 if torch.device(device).type == "cuda" else torch.bfloat16


def make_scaler(device: str | torch.device, amp: bool):
    """A GradScaler only where it helps: float16 on CUDA. Return None otherwise.

    float16 cannot hold gradients smaller than about 6e-8; the scaler multiplies the loss by a
    large factor before backward so they survive, then divides the gradients back before the step.
    bfloat16 (CPU) has float32's range, so no scaling is needed there.
    """
    raise NotImplementedError("Lesson 8, step 4: torch.amp.GradScaler(\"cuda\") if amp and the device is cuda, else None")


def train_one_epoch_amp(model: nn.Module, loader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                        device: str | torch.device = "cpu", amp: bool = False, scaler=None) -> float:
    """Lesson 6's epoch with two additions: autocast around the forward pass, and a scaler when given.

    Compute the forward pass inside ``torch.autocast(device_type=device.type, dtype=autocast_dtype(device), enabled=amp)``,
    then the loss *outside* it in float32 (``model_output.float()``). With a scaler:
    ``scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()``; without: backward and step as usual.
    Use ``optimizer.zero_grad(set_to_none=True)``. Return the mean loss weighted by batch size.
    """
    raise NotImplementedError("Lesson 8, step 4: the Lesson 6 loop with autocast and the optional scaler")


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, history: dict, best_val: float) -> None:
    """Everything needed to continue training as if nothing happened: model, optimizer, epoch, history, best_val."""
    raise NotImplementedError("Lesson 8, step 6: torch.save({\"model\": ..., \"optimizer\": ..., \"epoch\": epoch, \"history\": history, \"best_val\": best_val}, path)")


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, device: str | torch.device = "cpu") -> dict:
    """Load a checkpoint into ``model`` (and ``optimizer`` if given) and return the whole dict.

    Use ``torch.load(path, map_location=device, weights_only=True)``: a checkpoint saved on a GPU must open on a CPU.
    """
    raise NotImplementedError("Lesson 8, step 6: ckpt = torch.load(...); model.load_state_dict(ckpt[\"model\"]); optimizer too; return ckpt")


def fit_v2(model: nn.Module, train_loader, val_loader, epochs: int = 10, lr: float = 1e-3, device: str | torch.device = "cpu",
           amp: bool = False, loss_fn: nn.Module | None = None, weight_decay: float = 0.0, patience: int | None = None,
           ckpt_path: str | Path | None = None, resume: bool = False, log_dir: str | Path | None = None, seed: int = 0) -> dict:
    """Training loop v2. Lesson 6's fit_loader plus: device placement, optional autocast, a checkpoint
    after every epoch, resuming from it, per-epoch timing, and optional TensorBoard logging.

    The history gains an "epoch_seconds" list. Seed the RNG with ``seed + epoch`` at the start of
    every epoch, so a resumed run shuffles exactly like an uninterrupted one. When ``resume`` is True
    and ``ckpt_path`` exists, load it and continue from ``epoch + 1`` with its history.
    """
    raise NotImplementedError("Lesson 8, step 7: fit_loader with device, amp, checkpointing, resume, timing and logging")
