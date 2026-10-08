"""Lesson 2. Let PyTorch find the slopes, and go from one feature to many (reference solution)."""

from __future__ import annotations

import torch
from torch import Tensor


def autograd_gradients(x: Tensor, y: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Lesson 1's ``gradients``, computed by autograd."""
    pred = w * x + b                              # autograd records every operation from here on
    loss = ((pred - y) ** 2).mean()
    loss.backward()                               # fills w.grad and b.grad
    return w.grad.clone(), b.grad.clone()


def predict_many(X: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """(N, d) @ (d,) + () -> (N,)"""
    return X @ w + b


def fit_autograd(X: Tensor, y: Tensor, lr: float = 0.005, steps: int = 4000):
    """Gradient descent with autograd doing the slopes."""
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    losses = []
    for _ in range(steps):
        loss = ((predict_many(X, w, b) - y) ** 2).mean()
        losses.append(loss.item())
        loss.backward()                           # w.grad and b.grad now hold the slopes
        with torch.no_grad():                     # updates are bookkeeping, not part of the model
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                            # otherwise the next backward() adds to these
        b.grad.zero_()
    return w.detach(), b.detach(), losses
