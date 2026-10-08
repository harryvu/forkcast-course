"""Lesson 9 definition of done.

Run:  pytest tests/test_09.py -q
Expected: 7 passed.   (Uses random ResNet weights so it never downloads; the notebook uses the pretrained ones.)
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module

transfer = lesson_module("transfer", "build_transfer_model", "freeze", "param_groups", "top_k_accuracy", "confusion_matrix", lesson="Lesson 9")
vision = lesson_module("vision", "shape_loaders", lesson="Lesson 7")
train = lesson_module("train", "fit_v2", lesson="Lesson 8")
count_parameters = lesson_module("model", "count_parameters", lesson="Lesson 4").count_parameters

fast = pytest.mark.fast


@fast
def test_freeze_switches_every_gradient_off():
    model = freeze_target = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    assert count_parameters(model) == 58
    assert transfer.freeze(freeze_target) is freeze_target, "return the module so calls can be chained"
    assert count_parameters(model) == 0 and all(not p.requires_grad for p in model.parameters())


@fast
def test_transfer_model_has_a_fresh_head_on_a_frozen_backbone():
    torch.manual_seed(0)
    model = transfer.build_transfer_model(n_classes=7, pretrained=False)
    assert isinstance(model.fc, nn.Linear) and model.fc.in_features == 512 and model.fc.out_features == 7
    assert count_parameters(model) == 512 * 7 + 7, "only the new head should be trainable"
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("fc."))
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 7)


@fast
def test_trainable_blocks_unfreeze_from_the_top():
    torch.manual_seed(0)
    model = transfer.build_transfer_model(n_classes=7, pretrained=False, trainable_blocks=1)
    assert all(p.requires_grad for p in model.layer4.parameters()), "trainable_blocks=1 means layer4 trains"
    assert all(not p.requires_grad for p in model.layer3.parameters()), "layer3 stays frozen"
    assert count_parameters(model) == 512 * 7 + 7 + sum(p.numel() for p in model.layer4.parameters())


@fast
def test_param_groups_give_the_head_and_backbone_their_own_learning_rates():
    torch.manual_seed(0)
    model = transfer.build_transfer_model(n_classes=7, pretrained=False, trainable_blocks=1)
    groups = transfer.param_groups(model, lr_head=1e-3, lr_backbone=1e-4)
    assert [g["lr"] for g in groups] == [1e-3, 1e-4]
    head_ids = {id(p) for p in model.fc.parameters()}
    assert {id(p) for p in groups[0]["params"]} == head_ids
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert {id(p) for g in groups for p in g["params"]} == trainable_ids, "every trainable parameter in exactly one group, no frozen ones"
    torch.optim.Adam(groups)                                   # must be accepted by an optimizer


@fast
def test_top_k_accuracy():
    logits = torch.tensor([[0.1, 0.5, 0.3, 0.9], [2.0, 1.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.9], [1.0, 0.9, 0.8, 0.7]])
    y = torch.tensor([3, 1, 3, 3])
    assert abs(transfer.top_k_accuracy(logits, y, k=1) - 0.25) < 1e-6
    assert abs(transfer.top_k_accuracy(logits, y, k=2) - 0.75) < 1e-6
    assert abs(transfer.top_k_accuracy(logits, y, k=4) - 1.0) < 1e-6


@fast
def test_confusion_matrix_counts_true_rows_and_predicted_columns():
    y = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    preds = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0, 2])
    cm = transfer.confusion_matrix(preds, y, n_classes=3)
    assert cm.shape == (3, 3) and cm.dtype == torch.long
    assert torch.equal(cm, torch.tensor([[2, 1, 0], [0, 1, 1], [1, 0, 3]]))
    assert torch.equal(cm.sum(dim=1), torch.bincount(y)), "rows sum to the number of true images per class"
    assert torch.allclose(transfer.per_class_accuracy(cm), torch.tensor([2 / 3, 1 / 2, 3 / 4]))
    assert transfer.most_confused(cm, k=1) == [(0, 1, 1)] or transfer.most_confused(cm, k=1) == [(1, 2, 1)] or transfer.most_confused(cm, k=1) == [(2, 0, 1)]


@fast
def test_only_the_unfrozen_parameters_change_when_training():
    train_loader, val_loader = vision.shape_loaders(n_train=320, n_val=64, batch_size=32)
    torch.manual_seed(0)
    model = transfer.build_transfer_model(n_classes=4, pretrained=False, trainable_blocks=0)
    conv_before = model.conv1.weight.detach().clone()
    fc_before = model.fc.weight.detach().clone()
    history = train.fit_v2(model, train_loader, val_loader, epochs=2, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
    assert torch.equal(model.conv1.weight, conv_before), "a frozen layer must not move"
    assert not torch.equal(model.fc.weight, fc_before), "the head must learn"
    assert history["train"][-1] < history["train"][0]
