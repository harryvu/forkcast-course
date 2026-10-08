"""Lesson 2. Let PyTorch find the slopes, and go from one feature to many.

``X`` has shape (N, d): one row per order, one column per feature.
``w`` has shape (d,): one weight per feature. ``b`` is a single number, shape ().
Replace every ``raise NotImplementedError`` with a few lines of code.
"""

from __future__ import annotations

import torch
from torch import Tensor


def autograd_gradients(x: Tensor, y: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Lesson 1's ``gradients``, but computed by autograd instead of by hand.

    ``w`` and ``b`` are 0-d tensors created with ``requires_grad=True``.
    Compute the mean squared error of ``w * x + b`` against ``y``, call
    ``.backward()`` on it, and return copies of ``w.grad`` and ``b.grad``.
    """
    raise NotImplementedError("Lesson 2, step 3: loss = ...; loss.backward(); return w.grad.clone(), b.grad.clone()")


def predict_many(X: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """A weighted sum of every feature, for every order at once.

    (N, d) @ (d,) + () -> (N,)
    """
    raise NotImplementedError("Lesson 2, step 5: X @ w + b")


def fit_autograd(X: Tensor, y: Tensor, lr: float = 0.005, steps: int = 4000):
    """Gradient descent on w (shape (d,)) and b (shape ()), with autograd doing the slopes.

    Start w at zeros and b at zero, both with ``requires_grad=True``. Each step:
    predict, compute the mean squared error, record ``loss.item()``, call
    ``loss.backward()``, update w and b inside ``torch.no_grad()``, then zero
    both ``.grad`` tensors. Return ``(w.detach(), b.detach(), losses)``.
    """
    raise NotImplementedError("Lesson 2, step 6: the training loop with autograd")
