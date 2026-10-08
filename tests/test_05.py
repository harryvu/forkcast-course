"""Lesson 5 definition of done.

Run:  pytest tests/test_05.py -q
Expected: 7 passed.
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module
from forkcast.data import make_orders

train = lesson_module("train", "fit_with_validation", lesson="Lesson 5")
MLP = lesson_module("model", "MLP", lesson="Lesson 4").MLP
Normalizer = lesson_module("preprocess", "Normalizer", lesson="Lesson 3").Normalizer
train_val_test_split = lesson_module("split", "train_val_test_split", lesson="Lesson 5").train_val_test_split

fast = pytest.mark.fast


def _small_problem(n_train=40):
    """A big model's favourite trap: very little training data, a fixed validation set."""
    X, y = make_orders(2000, seed=0, nonlinear=True)
    (X_tr, y_tr), (X_val, y_val), _ = train_val_test_split(X, y, seed=0)
    norm = Normalizer().fit(X_tr[:n_train])
    return norm.transform(X_tr[:n_train]), y_tr[:n_train], norm.transform(X_val), y_val


@fast
def test_split_is_shuffled_disjoint_and_complete():
    X, y = make_orders(2000, seed=0, nonlinear=True)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = train_val_test_split(X, y, seed=0)
    assert (len(X_tr), len(X_val), len(X_te)) == (1400, 300, 300)
    assert len(y_tr) == 1400 and len(y_val) == 300 and len(y_te) == 300
    rows = torch.cat([X_tr, X_val, X_te])
    assert rows.shape == X.shape
    assert torch.equal(rows.sort(dim=0).values, X.sort(dim=0).values), "every row must appear exactly once across the three parts"
    assert not torch.equal(X_tr, X[:1400]), "shuffle before cutting; the first 70% of rows is not a random sample"
    (X_tr2, _), _, _ = train_val_test_split(X, y, seed=0)
    assert torch.equal(X_tr, X_tr2), "the same seed must give the same split"


@fast
def test_dropout_layers_exist_only_when_asked():
    plain = MLP(4, (32, 32))
    assert not any(isinstance(m, nn.Dropout) for m in plain.net), "dropout=0 must add nothing (Lesson 4's model must be unchanged)"
    reg = MLP(4, (32, 32), dropout=0.2)
    kinds = [type(m) for m in reg.net]
    assert kinds == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear], [k.__name__ for k in kinds]
    assert reg.net[2].p == 0.2


@fast
def test_dropout_is_random_in_train_mode_and_off_in_eval_mode():
    model = MLP(4, (64,), dropout=0.5)
    X, _ = make_orders(8, seed=0)
    model.train()
    assert not torch.equal(model(X), model(X)), "in training mode dropout should change the output from call to call"
    model.eval()
    assert torch.equal(model(X), model(X)), "in eval mode dropout must be switched off"


@fast
def test_big_model_little_data_overfits():
    X_tr, y_tr, X_val, y_val = _small_problem()
    torch.manual_seed(0)
    history = train.fit_with_validation(MLP(4, (64, 64)), X_tr, y_tr, X_val, y_val, epochs=800, lr=0.01)
    assert len(history["train"]) == 800 and len(history["val"]) == 800
    assert history["train"][-1] < 2.0, "40 points, 4545 parameters: the training loss should collapse below the noise"
    assert history["val"][-1] > 4 * history["train"][-1], f"validation {history['val'][-1]:.1f} should be far above training {history['train'][-1]:.2f}: that gap is overfitting"


@fast
def test_early_stopping_stops_and_keeps_the_best_weights():
    X_tr, y_tr, X_val, y_val = _small_problem()
    torch.manual_seed(0)
    model = MLP(4, (64, 64))
    history = train.fit_with_validation(model, X_tr, y_tr, X_val, y_val, epochs=800, lr=0.01, patience=30)
    assert len(history["val"]) < 800, "with patience=30 training should stop well before 800 epochs"
    assert history["best_val"] == min(history["val"])
    assert history["val"][history["best_epoch"]] == history["best_val"]
    assert abs(train.evaluate(model, X_val, y_val, nn.MSELoss()) - history["best_val"]) < 1e-4, "the returned model must carry the best weights, not the last ones"


@fast
def test_weight_decay_and_dropout_shrink_the_gap():
    X_tr, y_tr, X_val, y_val = _small_problem()
    torch.manual_seed(0)
    plain = train.fit_with_validation(MLP(4, (64, 64)), X_tr, y_tr, X_val, y_val, epochs=800, lr=0.01)
    torch.manual_seed(0)
    reg = train.fit_with_validation(MLP(4, (64, 64), dropout=0.2), X_tr, y_tr, X_val, y_val, epochs=800, lr=0.01, weight_decay=1e-2)
    assert reg["best_val"] < 0.8 * plain["val"][-1], f"regularized best validation {reg['best_val']:.1f} should clearly beat the unregularized final {plain['val'][-1]:.1f}"


@fast
def test_fit_with_validation_passes_weight_decay_to_the_optimizer(monkeypatch):
    seen = {}
    class Spy(torch.optim.Adam):
        def __init__(self, params, lr, weight_decay=0.0):
            seen["weight_decay"] = weight_decay
            super().__init__(params, lr=lr, weight_decay=weight_decay)
    X_tr, y_tr, X_val, y_val = _small_problem()
    train.fit_with_validation(MLP(4, (8,)), X_tr, y_tr, X_val, y_val, epochs=2, lr=0.01, optimizer_cls=Spy, weight_decay=0.05)
    assert seen["weight_decay"] == 0.05
