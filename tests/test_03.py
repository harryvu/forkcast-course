"""Lesson 3 definition of done.

Run:  pytest tests/test_03.py -q
Expected: 7 passed.
"""

import math

import pytest
import torch
from torch import nn

from conftest import lesson_module
from forkcast.data import TRUE_BIAS, TRUE_WEIGHTS, make_orders

train = lesson_module("train", "fit", "save", "load", lesson="Lesson 3")
LinearRegressor = lesson_module("model", "LinearRegressor", lesson="Lesson 3").LinearRegressor
Normalizer = lesson_module("preprocess", "Normalizer", lesson="Lesson 3").Normalizer

fast = pytest.mark.fast


@fast
def test_normalizer_uses_training_statistics_only():
    X, _ = make_orders(400, seed=0)
    other, _ = make_orders(50, seed=9)
    norm = Normalizer().fit(X)
    Xn = norm.transform(X)
    assert Xn.shape == X.shape
    assert torch.allclose(Xn.mean(dim=0), torch.zeros(4), atol=1e-5), "training columns should have mean 0"
    assert torch.allclose(Xn.std(dim=0, correction=0), torch.ones(4), atol=1e-4), "training columns should have std 1"
    expected = (other - X.mean(dim=0)) / X.std(dim=0, correction=0)
    assert torch.allclose(norm.transform(other), expected, atol=1e-5), "transform must reuse the stored mean and std, not recompute them"
    assert Normalizer().fit_transform(X).shape == X.shape


@fast
def test_model_shapes_and_parameters():
    model = LinearRegressor(4)
    X, _ = make_orders(10, seed=0)
    out = model(X)
    assert out.shape == (10,), f"forward must return (N,), got {tuple(out.shape)}: did you squeeze the last dimension?"
    assert model.linear.weight.shape == (1, 4) and model.linear.bias.shape == (1,)
    assert sum(p.numel() for p in model.parameters()) == 5


@fast
def test_model_is_the_formula_from_lesson_2():
    model = LinearRegressor(4)
    X, _ = make_orders(10, seed=0)
    by_hand = X @ model.linear.weight[0] + model.linear.bias[0]
    assert torch.allclose(model(X), by_hand, atol=1e-5)


@fast
def test_train_one_epoch_takes_one_step_down():
    X, y = make_orders(400, seed=0)
    Xn = Normalizer().fit_transform(X)
    model = LinearRegressor(4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = train.evaluate(model, Xn, y, nn.MSELoss())
    returned = train.train_one_epoch(model, Xn, y, opt, nn.MSELoss())
    after = train.evaluate(model, Xn, y, nn.MSELoss())
    assert isinstance(returned, float) and isinstance(before, float)
    assert math.isclose(returned, before, rel_tol=1e-5), "train_one_epoch should return the loss measured before the step"
    assert after < before, "one step must reduce the loss"


@fast
def test_fit_on_normalized_features_recovers_the_true_rule():
    X, y = make_orders(400, seed=0)
    norm = Normalizer().fit(X)
    model = LinearRegressor(4)
    losses = train.fit(model, norm.transform(X), y, epochs=300, lr=0.1)
    assert len(losses) == 300 and losses[-1] < 5.0, f"loss should reach the noise floor (~4), got {losses[-1]:.2f}"
    w_norm, b_norm = model.linear.weight[0].detach(), model.linear.bias[0].detach()
    w_orig = w_norm / norm.std                           # undo the scaling to compare with the generator
    b_orig = b_norm - (w_norm * norm.mean / norm.std).sum()
    assert torch.allclose(w_orig, TRUE_WEIGHTS, atol=0.5), f"original-scale weights {w_orig.tolist()} vs truth {TRUE_WEIGHTS.tolist()}"
    assert abs(b_orig.item() - TRUE_BIAS) < 1.0


@fast
def test_same_learning_rate_fails_on_raw_features():
    X, y = make_orders(400, seed=0)
    losses = train.fit(LinearRegressor(4), X, y, epochs=60, lr=0.1)
    assert not math.isfinite(losses[-1]) or losses[-1] > losses[0], "raw features with lr=0.1 should diverge; that is why Lesson 3 normalizes"


@fast
def test_save_and_load_give_identical_predictions(tmp_path):
    X, y = make_orders(50, seed=0)
    model = LinearRegressor(4)
    train.fit(model, Normalizer().fit_transform(X), y, epochs=5, lr=0.1)
    path = tmp_path / "linear.pt"
    train.save(model, path)
    assert path.exists()
    fresh = train.load(LinearRegressor(4), path)
    assert torch.equal(fresh(X), model(X)), "the reloaded model must give exactly the same predictions"
    assert set(model.state_dict().keys()) == {"linear.weight", "linear.bias"}
