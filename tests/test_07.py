"""Lesson 7 definition of done.

Run:  pytest tests/test_07.py -q
Expected: 7 passed.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from conftest import lesson_module

vision = lesson_module("vision", "SmallCNN", "conv_output_size", "accuracy", "cross_entropy_by_hand", lesson="Lesson 7")
train = lesson_module("train", "fit_loader", lesson="Lesson 6")
count_parameters = lesson_module("model", "count_parameters", lesson="Lesson 4").count_parameters

fast = pytest.mark.fast


@fast
def test_make_shapes():
    x, y = vision.make_shapes(200, seed=0)
    assert x.shape == (200, 3, 32, 32) and x.dtype == torch.float32 and y.shape == (200,) and y.dtype == torch.long
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert all((y == k).sum() >= 20 for k in range(4)), "all four shapes should appear"
    x2, _ = vision.make_shapes(200, seed=0)
    assert torch.equal(x, x2)


@fast
def test_conv_output_size_matches_pytorch():
    for size, k, s, p in [(32, 3, 1, 1), (32, 3, 2, 1), (32, 5, 1, 0), (28, 3, 1, 0), (64, 7, 2, 3), (32, 2, 2, 0)]:
        real = nn.Conv2d(1, 1, kernel_size=k, stride=s, padding=p)(torch.zeros(1, 1, size, size)).shape[-1]
        assert vision.conv_output_size(size, k, s, p) == real, f"size={size} k={k} s={s} p={p}: formula {vision.conv_output_size(size, k, s, p)}, PyTorch {real}"


@fast
def test_small_cnn_structure_and_shapes():
    model = vision.SmallCNN(n_classes=4, size=32)
    x, _ = vision.make_shapes(8, seed=0)
    assert model(x).shape == (8, 4), "logits: one row per image, one column per class"
    kinds = [type(m) for m in model.features]
    assert kinds[:4] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d], [k.__name__ for k in kinds[:4]]
    assert model.features(x).shape == (8, 32, 8, 8), "two pools halve 32 twice"
    expected = (3 * 3 * 3 * 16 + 16) + 2 * 16 + (3 * 3 * 16 * 32 + 32) + 2 * 32 + (32 * 8 * 8 * 64 + 64) + (64 * 4 + 4)
    assert count_parameters(model) == expected, f"expected {expected} parameters, got {count_parameters(model)}"
    assert vision.SmallCNN(n_classes=10, size=64)(torch.zeros(2, 3, 64, 64)).shape == (2, 10), "size and n_classes must both be respected"


@fast
def test_accuracy():
    logits = torch.tensor([[2.0, 0.1, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.4], [1.0, 0.0, 0.0, 0.0]])
    y = torch.tensor([0, 1, 2, 3])
    assert abs(vision.accuracy(logits, y) - 0.75) < 1e-6
    assert isinstance(vision.accuracy(logits, y), float)


@fast
def test_cross_entropy_by_hand_matches_pytorch():
    torch.manual_seed(0)
    logits, y = torch.randn(16, 4) * 3, torch.randint(0, 4, (16,))
    assert torch.allclose(vision.cross_entropy_by_hand(logits, y), nn.CrossEntropyLoss()(logits, y), atol=1e-5)
    sure_and_right = torch.tensor([[10.0, 0.0, 0.0, 0.0]]); sure_and_wrong = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
    assert vision.cross_entropy_by_hand(sure_and_right, torch.tensor([0])) < 0.01
    assert vision.cross_entropy_by_hand(sure_and_wrong, torch.tensor([0])) > 9.0, "confident and wrong must be punished hard"


@fast
def test_training_loss_falls_over_the_first_epochs():
    train_loader, val_loader = vision.shape_loaders(n_train=640, n_val=128)
    torch.manual_seed(0)
    history = train.fit_loader(vision.SmallCNN(), train_loader, val_loader, epochs=3, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
    chance = torch.log(torch.tensor(4.0)).item()          # guessing uniformly over four classes
    assert history["train"][0] < chance * 1.1, "even the first epoch should beat random guessing on average"
    assert history["train"][-1] < history["train"][0], "the training loss must fall over three epochs"


@fast
def test_small_cnn_learns_the_shapes():
    train_loader, val_loader = vision.shape_loaders(n_train=1000, n_val=400)
    torch.manual_seed(0)
    model = vision.SmallCNN()
    train.fit_loader(model, train_loader, val_loader, epochs=6, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
    acc = vision.accuracy_loader(model, val_loader)
    assert acc > 0.85, f"six epochs on 1000 synthetic shapes should classify well: accuracy {acc:.2f}"
