"""Lesson 1. Linear regression by hand: predict, measure, correct, repeat (reference solution).

The model is ``minutes = w * distance + b``. ``w`` and ``b`` are 0-dimensional
tensors (single numbers). ``x`` and ``y`` have shape (N,).
"""

from __future__ import annotations

import torch
from torch import Tensor


def predict(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """The model. (N,) -> (N,): one predicted minute count per delivery."""
    return w * x + b                       # () * (N,) + () broadcasts to (N,)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error: one number (shape ()) saying how wrong ``pred`` is."""
    return ((pred - target) ** 2).mean()


def gradients(x: Tensor, y: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Slopes of the loss with respect to w and b, each shape ()."""
    e = predict(x, w, b) - y               # (N,) errors
    dw = 2 * (e * x).mean()
    db = 2 * e.mean()
    return dw, db


def gradient_step(x: Tensor, y: Tensor, w: Tensor, b: Tensor, lr: float) -> tuple[Tensor, Tensor]:
    """One step downhill: move w and b against their slopes by ``lr`` times the slope."""
    dw, db = gradients(x, y, w, b)
    return w - lr * dw, b - lr * db


def fit(x: Tensor, y: Tensor, lr: float = 0.01, steps: int = 3000, w0: float = 0.0, b0: float = 0.0):
    """Run gradient descent from (w0, b0). Returns ``(w, b, losses)``."""
    w = torch.tensor(float(w0))
    b = torch.tensor(float(b0))
    losses = []
    for _ in range(steps):
        losses.append(mse_loss(predict(x, w, b), y).item())
        w, b = gradient_step(x, y, w, b, lr)
    return w, b, losses
