"""Lesson 7. Images as tensors, and the first convolutional network.

``make_shapes`` (provided) draws synthetic pictures of four shapes so the tests
run offline; the notebook trains the same network on real dish photos.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

SHAPE_CLASSES = ["circle", "square", "triangle", "cross"]


def make_shapes(n: int = 1000, seed: int = 0, size: int = 32) -> tuple[Tensor, Tensor]:
    """Return ``(images, labels)``: images (n, 3, size, size) float32 in [0, 1], labels (n,) int64.

    Each image is one coloured shape on a dark, lightly noisy background, at a random
    position and scale. Classes in SHAPE_CLASSES order: circle, square, triangle, cross.
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n,), generator=g)
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
    images = torch.zeros(n, 3, size, size)
    for i in range(n):
        cx, cy = (size * (0.3 + 0.4 * torch.rand(2, generator=g))).tolist()
        r = size * (0.12 + 0.13 * torch.rand(1, generator=g).item())
        dx, dy = xx - cx, yy - cy
        k = labels[i].item()
        if k == 0:
            mask = dx ** 2 + dy ** 2 <= r ** 2
        elif k == 1:
            mask = (dx.abs() <= r) & (dy.abs() <= r)
        elif k == 2:
            mask = (dy >= -r) & (dy <= r) & (dx.abs() <= (r - dy) * 0.6)
        else:
            mask = ((dx.abs() <= r * 0.3) & (dy.abs() <= r)) | ((dy.abs() <= r * 0.3) & (dx.abs() <= r))
        colour = 0.5 + 0.5 * torch.rand(3, generator=g)
        images[i] = 0.08 * torch.rand(3, size, size, generator=g)
        images[i][:, mask] = colour[:, None]
    return images.clamp(0, 1), labels


def shape_loaders(n_train: int = 1000, n_val: int = 400, batch_size: int = 64, size: int = 32) -> tuple[DataLoader, DataLoader]:
    """Loaders over TensorDatasets of synthetic shapes (train seed 0, validation seed 1)."""
    x_tr, y_tr = make_shapes(n_train, seed=0, size=size)
    x_val, y_val = make_shapes(n_val, seed=1, size=size)
    return (DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False))


def conv_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Spatial size after a convolution or pooling layer: floor((size + 2p - k) / s) + 1."""
    return (size + 2 * padding - kernel) // stride + 1


def accuracy(logits: Tensor, y: Tensor) -> float:
    """Fraction of rows whose highest logit is the true class. (N, C), (N,) -> float."""
    return (logits.argmax(dim=1) == y).float().mean().item()


def cross_entropy_by_hand(logits: Tensor, y: Tensor) -> Tensor:
    """What nn.CrossEntropyLoss computes: minus the log-probability of the true class, averaged.

    Use F.log_softmax over the class dimension, pick each row's true-class entry, negate, mean.
    """
    logp = F.log_softmax(logits, dim=1)                       # (N, C) log-probabilities, rows sum to 1 in probability
    return -logp[torch.arange(len(y)), y].mean()               # pick each row's true class


class SmallCNN(nn.Module):
    """Two conv blocks, then a small classifier head.

    Block: Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2). Channels 3 -> 16 -> 32.
    After two pools a size x size image is (size/4) x (size/4), so the head's first
    Linear takes 32 * (size // 4) ** 2 inputs. The head ends in Linear(64, n_classes), no activation:
    those are the logits.
    """

    def __init__(self, n_classes: int = 4, size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # (N, 16, size/2, size/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # (N, 32, size/4, size/4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),                                          # (N, 32 * (size/4)^2)
            nn.Linear(32 * (size // 4) ** 2, 64), nn.ReLU(),
            nn.Linear(64, n_classes),                              # logits, no activation
        )

    def forward(self, x: Tensor) -> Tensor:
        """(N, 3, size, size) -> (N, n_classes)"""
        return self.head(self.features(x))


@torch.no_grad()
def accuracy_loader(model: nn.Module, loader, device: str | torch.device = "cpu") -> float:
    """Accuracy over every batch in ``loader``."""
    model.eval()
    correct, seen = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        seen += len(y)
    return correct / seen


# ---- Lesson 8: data augmentation --------------------------------------------

def train_transform(size: int = 32):
    """Random changes that keep the label true: a flipped, slightly re-framed, re-lit dish is the same dish."""
    from torchvision.transforms import v2
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class AugmentedTensorDataset(Dataset):
    """Like TensorDataset(images, labels) but applies ``transform`` to the image every time an item is fetched.

    A fresh random transform per fetch means the model never sees exactly the same picture twice.
    With ``transform=None`` it must return items identical to the stored tensors.
    """

    def __init__(self, images: Tensor, labels: Tensor, transform=None):
        raise NotImplementedError("Lesson 8, step 5: store images, labels, transform")

    def __len__(self) -> int:
        raise NotImplementedError("Lesson 8, step 5")

    def __getitem__(self, i: int):
        raise NotImplementedError("Lesson 8, step 5: x = self.images[i]; apply the transform if there is one; return x, self.labels[i]")


def augmented_shape_loaders(n_train: int = 1000, n_val: int = 400, batch_size: int = 64, size: int = 32) -> tuple[DataLoader, DataLoader]:
    """Training loader with augmentation, validation loader without (never augment what you evaluate on)."""
    x_tr, y_tr = make_shapes(n_train, seed=0, size=size)
    x_val, y_val = make_shapes(n_val, seed=1, size=size)
    return (DataLoader(AugmentedTensorDataset(x_tr, y_tr, train_transform(size)), batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False))
