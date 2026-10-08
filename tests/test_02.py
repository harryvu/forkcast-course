"""Lesson 2 definition of done.

Run:  pytest tests/test_02.py -q
Expected: 6 passed.
"""

import pytest
import torch

from conftest import lesson_module

linear = lesson_module("linear", "gradients", lesson="Lesson 1")
regression = lesson_module("regression", "fit_autograd", lesson="Lesson 2")
from forkcast.data import FEATURES, TRUE_BIAS, TRUE_WEIGHTS, make_deliveries, make_orders

fast = pytest.mark.fast


@fast
def test_make_orders():
    X, y = make_orders(400, seed=0)
    assert X.shape == (400, 4) and y.shape == (400,)
    assert X.dtype == torch.float32 and len(FEATURES) == 4
    assert torch.equal(X[:, 1], X[:, 1].round()) and X[:, 1].min() >= 1 and X[:, 1].max() <= 6, "items are whole numbers 1 to 6"
    assert set(X[:, 2].unique().tolist()) <= {0.0, 1.0} and set(X[:, 3].unique().tolist()) <= {0.0, 1.0}, "rain and rush hour are 0 or 1"
    X2, _ = make_orders(400, seed=0)
    assert torch.equal(X, X2), "the same seed must give the same orders"


@fast
def test_autograd_gradients_match_lesson_1():
    x, y = make_deliveries(50, seed=1)
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    dw, db = regression.autograd_gradients(x, y, w, b)
    dw_hand, db_hand = linear.gradients(x, y, torch.tensor(1.0), torch.tensor(2.0))
    assert torch.allclose(dw, dw_hand, atol=1e-4), f"autograd dw {dw.item():.4f} vs your Lesson 1 formula {dw_hand.item():.4f}"
    assert torch.allclose(db, db_hand, atol=1e-4), f"autograd db {db.item():.4f} vs your Lesson 1 formula {db_hand.item():.4f}"
    assert not dw.requires_grad, "return a plain copy (clone) of the gradient, not the live .grad tensor"


@fast
def test_predict_many_is_a_weighted_sum():
    X, _ = make_orders(10, seed=3)
    w = torch.tensor([3.5, 1.5, 4.0, 6.0])
    b = torch.tensor(8.0)
    out = regression.predict_many(X, w, b)
    assert out.shape == (10,), f"expected shape (10,), got {tuple(out.shape)}"
    by_hand = torch.stack([(X[i] * w).sum() + b for i in range(10)])
    assert torch.allclose(out, by_hand, atol=1e-5), "X @ w + b must equal the weighted sum of each row's features"


@fast
def test_predict_many_handles_one_feature():
    x, _ = make_deliveries(5, seed=0)
    out = regression.predict_many(x[:, None], torch.tensor([3.5]), torch.tensor(8.0))
    assert out.shape == (5,)
    assert torch.allclose(out, 3.5 * x + 8.0)


@fast
def test_fit_autograd_recovers_the_true_rule():
    X, y = make_orders(400, seed=0)
    w, b, losses = regression.fit_autograd(X, y, lr=0.005, steps=4000)
    assert w.shape == (4,) and b.shape == ()
    assert not w.requires_grad and not b.requires_grad, "return detached tensors; the caller does not want the autograd history"
    assert len(losses) == 4000 and all(isinstance(v, float) for v in losses)
    assert losses[-1] < 0.05 * losses[0], f"loss should collapse: went from {losses[0]:.1f} to {losses[-1]:.2f}"
    assert torch.allclose(w, TRUE_WEIGHTS, atol=0.5), f"learned {w.tolist()}, data was made with {TRUE_WEIGHTS.tolist()}"
    assert abs(b.item() - TRUE_BIAS) < 1.0, f"b={b.item():.2f}, data was made with {TRUE_BIAS}"


@fast
def test_fit_autograd_zeroes_the_gradients():
    # If the gradients are not zeroed they accumulate, every step grows, and the fit blows up.
    X, y = make_orders(400, seed=0)
    _, _, losses = regression.fit_autograd(X, y, lr=0.005, steps=200)
    diffs = torch.tensor(losses).diff()
    assert (diffs <= 1e-3).all(), "the loss must never go up with this learning rate: did you call .grad.zero_() every step?"
