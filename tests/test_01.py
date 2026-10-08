"""Lesson 1 definition of done.

Run:  pytest tests/test_01.py -q
Expected: 7 passed.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import lesson_module

linear = lesson_module("linear", "fit", lesson="Lesson 1")
from forkcast.data import TRUE_INTERCEPT, TRUE_SLOPE, make_deliveries

fast = pytest.mark.fast


@fast
def test_make_deliveries():
    x, y = make_deliveries(200, seed=0)
    assert x.shape == (200,) and y.shape == (200,)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.min() >= 0.5 and x.max() <= 10.0
    x2, _ = make_deliveries(200, seed=0)
    assert torch.equal(x, x2), "the same seed must give the same deliveries"


@fast
def test_predict():
    x = torch.tensor([1.0, 2.0, 4.0])
    out = linear.predict(x, torch.tensor(3.0), torch.tensor(8.0))
    assert out.shape == (3,), f"expected shape (3,), got {tuple(out.shape)}"
    assert torch.allclose(out, torch.tensor([11.0, 14.0, 20.0]))


@fast
def test_mse_loss():
    pred = torch.tensor([11.0, 14.0, 20.0])
    target = torch.tensor([10.0, 15.0, 20.0])
    loss = linear.mse_loss(pred, target)
    assert loss.shape == (), f"the loss must be a single number, got shape {tuple(loss.shape)}"
    assert torch.allclose(loss, F.mse_loss(pred, target)), "must match PyTorch's mse_loss"


@fast
def test_gradients_match_autograd():
    x, y = make_deliveries(50, seed=1)
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    linear.mse_loss(w * x + b, y).backward()        # PyTorch's own answer; Lesson 2 explains this line
    dw, db = linear.gradients(x, y, w.detach(), b.detach())
    assert dw.shape == () and db.shape == ()
    assert torch.allclose(dw, w.grad, atol=1e-4), f"dL/dw: yours {dw.item():.4f}, autograd {w.grad.item():.4f}"
    assert torch.allclose(db, b.grad, atol=1e-4), f"dL/db: yours {db.item():.4f}, autograd {b.grad.item():.4f}"


@fast
def test_gradient_step_moves_against_the_slope():
    x, y = make_deliveries(50, seed=1)
    w, b = torch.tensor(0.0), torch.tensor(0.0)
    dw, db = linear.gradients(x, y, w, b)
    w1, b1 = linear.gradient_step(x, y, w, b, lr=0.01)
    assert torch.allclose(w1, w - 0.01 * dw) and torch.allclose(b1, b - 0.01 * db)
    assert linear.mse_loss(linear.predict(x, w1, b1), y) < linear.mse_loss(linear.predict(x, w, b), y), "one small step must reduce the loss"


@fast
def test_fit_recovers_the_true_rule():
    x, y = make_deliveries(200, seed=0)
    w, b, losses = linear.fit(x, y, lr=0.01, steps=3000)
    assert len(losses) == 3000 and all(isinstance(v, float) for v in losses), "record one Python float per step"
    diffs = torch.tensor(losses).diff()
    assert (diffs <= 1e-3).all(), "with lr=0.01 the loss must never go up (tiny float32 wobble near the bottom is allowed)"
    assert abs(w.item() - TRUE_SLOPE) < 0.2, f"w={w.item():.3f}, the data was made with {TRUE_SLOPE}"
    assert abs(b.item() - TRUE_INTERCEPT) < 0.8, f"b={b.item():.3f}, the data was made with {TRUE_INTERCEPT}"


@fast
def test_too_large_learning_rate_diverges():
    x, y = make_deliveries(200, seed=0)
    _, _, losses = linear.fit(x, y, lr=0.1, steps=60)
    last = losses[-1]
    assert (not math.isfinite(last)) or last > losses[0], "lr=0.1 overshoots: the loss should blow up, not shrink"
