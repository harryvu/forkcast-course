"""Lesson 1. Linear regression by hand: predict, measure, correct, repeat.

The model is ``minutes = w * distance + b``. ``w`` and ``b`` are 0-dimensional
tensors (single numbers). ``x`` and ``y`` have shape (N,). Replace every
``raise NotImplementedError`` with one to three lines of code.
"""

from __future__ import annotations

import torch
from torch import Tensor


def predict(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """The model. (N,) -> (N,): one predicted minute count per delivery."""
    raise NotImplementedError("Lesson 1, step 3: w * x + b")


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error: one number (shape ()) saying how wrong ``pred`` is."""
    raise NotImplementedError("Lesson 1, step 3: square the errors, take the mean")


def gradients(x: Tensor, y: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Slopes of the loss with respect to w and b, each shape ().

    With e = predict(x, w, b) - y:
        dL/dw = 2 * mean(e * x)
        dL/db = 2 * mean(e)
    """
    raise NotImplementedError("Lesson 1, step 4: the two formulas above")


def gradient_step(x: Tensor, y: Tensor, w: Tensor, b: Tensor, lr: float) -> tuple[Tensor, Tensor]:
    """One step downhill: move w and b against their slopes by ``lr`` times the slope."""
    raise NotImplementedError("Lesson 1, step 5: new w = w - lr * dw, same for b")


def fit(x: Tensor, y: Tensor, lr: float = 0.01, steps: int = 3000, w0: float = 0.0, b0: float = 0.0):
    """Run gradient descent from (w0, b0).

    Returns ``(w, b, losses)`` where ``losses`` is a list of Python floats,
    one per step, recorded *before* each step (use ``.item()``).
    """
    raise NotImplementedError("Lesson 1, step 5: loop `steps` times: record loss, take a gradient_step")
