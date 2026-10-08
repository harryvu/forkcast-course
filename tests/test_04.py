"""Lesson 4 definition of done.

Run:  pytest tests/test_04.py -q
Expected: 7 passed.
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module
from forkcast.data import make_orders

train = lesson_module("train", "fit", lesson="Lesson 3")
_model = lesson_module("model", "MLP", "count_parameters", lesson="Lesson 4")
MLP, LinearRegressor, count_parameters = _model.MLP, _model.LinearRegressor, _model.count_parameters
Normalizer = lesson_module("preprocess", "Normalizer", lesson="Lesson 3").Normalizer

fast = pytest.mark.fast


def _data():
    X, y = make_orders(2000, seed=0, nonlinear=True)
    return Normalizer().fit_transform(X), y


@fast
def test_nonlinear_orders_keep_the_old_generator_intact():
    X_lin, y_lin = make_orders(50, seed=0)
    X_old, y_old = make_orders(50, seed=0, nonlinear=False)
    assert torch.equal(X_lin, X_old) and torch.equal(y_lin, y_old), "the default must stay exactly what Lessons 2 and 3 used"
    X_nl, y_nl = make_orders(50, seed=0, nonlinear=True)
    assert torch.equal(X_nl, X_lin), "same features; only the rule that makes minutes changed"
    assert y_nl.shape == (50,) and not torch.equal(y_nl, y_lin)
    far = X_nl[:, 0] > 7
    assert (y_nl[far] - y_lin[far]).mean() > (y_nl[~far] - y_lin[~far]).mean(), "far deliveries gain the most minutes"


@fast
def test_count_parameters():
    assert count_parameters(LinearRegressor(4)) == 5
    assert count_parameters(nn.Linear(4, 32)) == 4 * 32 + 32


@fast
def test_mlp_structure():
    model = MLP(4, hidden=(32, 32))
    X, _ = make_orders(10, seed=0)
    assert model(X).shape == (10,), "forward must return (N,): squeeze the trailing 1"
    assert count_parameters(model) == (4 * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
    kinds = [type(m) for m in model.net]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear], f"expected Linear, ReLU, Linear, ReLU, Linear, got {[k.__name__ for k in kinds]}"


@fast
def test_mlp_with_no_hidden_layers_is_linear_regression():
    model = MLP(4, hidden=())
    assert count_parameters(model) == 5
    assert [type(m) for m in model.net] == [nn.Linear]


@fast
def test_linear_model_underfits_nonlinear_deliveries():
    Xn, y = _data()
    torch.manual_seed(0)
    losses = train.fit(LinearRegressor(4), Xn, y, epochs=300, lr=0.1)
    assert losses[-1] > 9.0, f"a straight line should miss the bends: loss {losses[-1]:.2f} is suspiciously low"


@fast
def test_relu_mlp_learns_the_bends():
    Xn, y = _data()
    torch.manual_seed(0)
    losses = train.fit(MLP(4, (32, 32)), Xn, y, epochs=300, lr=0.01, optimizer_cls=torch.optim.Adam)
    assert losses[-1] < 7.0, f"the MLP should get close to the noise floor (~4): got {losses[-1]:.2f}"


@fast
def test_without_an_activation_depth_buys_nothing():
    Xn, y = _data()
    torch.manual_seed(0)
    identity = train.fit(MLP(4, (32, 32), activation=nn.Identity), Xn, y, epochs=300, lr=0.01, optimizer_cls=torch.optim.Adam)
    torch.manual_seed(0)
    linear = train.fit(LinearRegressor(4), Xn, y, epochs=300, lr=0.1)
    assert abs(identity[-1] - linear[-1]) < 1.5, f"two Linear layers with nothing between them are one Linear: {identity[-1]:.2f} vs {linear[-1]:.2f}"
