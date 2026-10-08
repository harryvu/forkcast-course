"""Lesson 9. Transfer learning: start from a network that already sees, and honest evaluation.

``extract_features``, ``per_class_accuracy`` and ``most_confused`` are provided; the rest are exercises.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import models

BACKBONE_BLOCKS = ["layer4", "layer3", "layer2", "layer1"]   # from the top of the network downwards


def freeze(module: nn.Module) -> nn.Module:
    """Switch off gradients for every parameter in ``module`` and return it."""
    raise NotImplementedError("Lesson 9, step 3: for p in module.parameters(): p.requires_grad_(False)")


def build_transfer_model(n_classes: int, pretrained: bool = True, trainable_blocks: int = 0) -> nn.Module:
    """A ResNet-18 that classifies our dishes.

    1. ``models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)``
    2. freeze the whole network
    3. replace ``model.fc`` with a fresh ``nn.Linear(model.fc.in_features, n_classes)`` (new layers require grad by default)
    4. for the first ``trainable_blocks`` entries of BACKBONE_BLOCKS, switch gradients back on with ``requires_grad_(True)``
    """
    raise NotImplementedError("Lesson 9, step 3: the four steps in the docstring")


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float) -> list[dict]:
    """Two groups for the optimizer: the head (``model.fc``) at ``lr_head``, every other trainable parameter at ``lr_backbone``.

    Return ``[{"params": [...], "lr": lr_head}, {"params": [...], "lr": lr_backbone}]``, skipping frozen parameters.
    """
    raise NotImplementedError("Lesson 9, step 5: split model.named_parameters() by whether the name starts with \"fc.\"")


def top_k_accuracy(logits: Tensor, y: Tensor, k: int = 5) -> float:
    """Fraction of rows whose true class is among the k largest logits. (N, C), (N,) -> float."""
    raise NotImplementedError("Lesson 9, step 6: logits.topk(k, dim=1).indices == y[:, None], any over dim 1, mean")


def confusion_matrix(preds: Tensor, y: Tensor, n_classes: int) -> Tensor:
    """(N,), (N,) -> (n_classes, n_classes) int64: rows are true classes, columns are predictions.

    cm[i, j] counts images of class i predicted as class j. A loop over the pairs is fine;
    ``torch.bincount(y * n_classes + preds, minlength=n_classes ** 2).view(n_classes, n_classes)`` is the one-liner.
    """
    raise NotImplementedError("Lesson 9, step 6: the bincount one-liner, or a loop")


@torch.no_grad()
def extract_features(model: nn.Module, loader, device: str | torch.device = "cpu") -> tuple[Tensor, Tensor]:
    """Run the frozen backbone once over a loader and cache its 512-d features: the fast way to train a head."""
    model.eval().to(device)
    fc, model.fc = model.fc, nn.Identity()            # temporarily return the pooled features instead of logits
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).cpu())
        labels.append(y)
    model.fc = fc
    return torch.cat(feats), torch.cat(labels)


def per_class_accuracy(cm: Tensor) -> Tensor:
    """Diagonal over row sums: how often each true class is recognised."""
    return cm.diagonal().float() / cm.sum(dim=1).clamp_min(1).float()


def most_confused(cm: Tensor, k: int = 5) -> list[tuple[int, int, int]]:
    """The k largest off-diagonal entries as (true, predicted, count)."""
    off = cm.clone()
    off.fill_diagonal_(0)
    flat = off.flatten()
    idx = flat.argsort(descending=True)[:k]
    n = cm.shape[0]
    return [(int(i // n), int(i % n), int(flat[i])) for i in idx if flat[i] > 0]
