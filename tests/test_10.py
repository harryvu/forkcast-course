"""Lesson 10 definition of done (optional lesson).

Run:  pytest tests/test_10.py -q
Expected: 7 passed.
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module

debug = lesson_module("debug", "GradCAM", "misclassified", "class_weights", "expected_calibration_error", "record_activation", lesson="Lesson 10")
vision = lesson_module("vision", "SmallCNN", lesson="Lesson 7")

fast = pytest.mark.fast


@fast
def test_class_counts_and_weights():
    labels = torch.tensor([0] * 400 + [1] * 100)
    counts = debug.class_counts(labels, n_classes=3)
    assert torch.equal(counts, torch.tensor([400, 100, 0]))
    w = debug.class_weights(torch.tensor([400, 100]))
    assert torch.allclose(w, torch.tensor([0.625, 2.5])), f"inverse frequency with mean 1: got {w.tolist()}"
    assert torch.isclose((w * torch.tensor([400.0, 100.0])).sum(), torch.tensor(500.0)), "count × weight is the same for every class: each class carries equal total weight"
    w3 = debug.class_weights(counts)
    assert torch.isfinite(w3).all(), "an empty class must not divide by zero"


@fast
def test_weighted_loss_makes_rare_class_mistakes_expensive():
    logits = torch.tensor([[2.0, 0.0]] * 9 + [[2.0, 0.0]])          # always predicts class 0
    y = torch.tensor([0] * 9 + [1])                                    # one rare-class image, misclassified
    plain = nn.CrossEntropyLoss()(logits, y)
    weighted = nn.CrossEntropyLoss(weight=debug.class_weights(torch.tensor([9, 1])))(logits, y)
    assert weighted > 2 * plain, f"the rare mistake should dominate: weighted {weighted:.3f}, plain {plain:.3f}"


@fast
def test_misclassified_returns_wrong_rows_most_confident_first():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 1.0, 0.9], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.2, 0.1, 0.0]])
    y = torch.tensor([1, 1, 0, 1, 2])                                   # rows 0, 3, 4 are wrong
    picks = debug.misclassified(logits, y, k=10)
    assert [p[0] for p in picks] == [0, 3, 4], f"wrong rows, most confident first: got {[p[0] for p in picks]}"
    assert picks[0][1:3] == (1, 0) and 0.98 < picks[0][3] <= 1.0
    assert [p[0] for p in debug.misclassified(logits, y, k=2)] == [0, 3]


@fast
def test_record_activation_captures_the_layer_and_removes_its_hook():
    torch.manual_seed(0)
    model = vision.SmallCNN()
    x, _ = vision.make_shapes(4, seed=0)
    act = debug.record_activation(model, model.features[0], x)
    assert act.shape == (4, 16, 32, 32), "the first conv's output"
    assert len(model.features[0]._forward_hooks) == 0, "the hook must be removed after use"
    pooled = debug.record_activation(model, model.features[7], x)
    assert pooled.shape == (4, 32, 8, 8)


@fast
def test_gradcam_shape_range_and_cleanup():
    torch.manual_seed(0)
    model = vision.SmallCNN()
    x, y = vision.make_shapes(4, seed=0)
    cam = debug.GradCAM(model, model.features[6])          # the second block's ReLU output: (N, 32, 16, 16)
    heat = cam(x)
    assert heat.shape == (4, 16, 16), f"one heatmap per image on the layer's grid, got {tuple(heat.shape)}"
    assert heat.min() >= 0 and heat.max() <= 1 + 1e-6 and not heat.requires_grad
    maxes = heat.flatten(1).max(dim=1).values
    assert all(abs(m - 1) < 1e-5 or m == 0 for m in maxes.tolist()), "each heatmap is normalised to a maximum of 1 (or is all zero when nothing supports the class)"
    other = cam(x, class_idx=(y + 1) % 4)
    assert not torch.allclose(heat, other), "the heatmap depends on which class you ask about"
    cam.remove()
    assert len(model.features[6]._forward_hooks) == 0


@fast
def test_gradcam_works_on_a_frozen_model():
    torch.manual_seed(0)
    model = vision.SmallCNN()
    for p in model.parameters():
        p.requires_grad_(False)
    x, _ = vision.make_shapes(2, seed=1)
    heat = debug.GradCAM(model, model.features[6])(x)
    assert heat.shape == (2, 16, 16), "a frozen backbone (Lesson 9) must still give a heatmap"


@fast
def test_expected_calibration_error():
    torch.manual_seed(0)
    n = 2000
    # perfectly calibrated: 70% confident predictions are right 70% of the time
    probs = torch.zeros(n, 2); probs[:, 0] = 0.7; probs[:, 1] = 0.3
    y = (torch.rand(n) > 0.7).long()                                    # class 0 with probability 0.7
    assert debug.expected_calibration_error(probs, y) < 0.03
    # overconfident: 95% confident, right half the time
    probs2 = torch.zeros(n, 2); probs2[:, 0] = 0.95; probs2[:, 1] = 0.05
    y2 = torch.arange(n) % 2
    assert abs(debug.expected_calibration_error(probs2, y2) - 0.45) < 0.01
