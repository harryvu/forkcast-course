"""Lesson 0. Six small tensor functions (reference solution).

Each docstring states the exact input and output shapes the tests expect.
Three of these come back later in the course: normalize_columns (Lesson 3),
flatten_images (Lesson 7) and pairwise_distance (Lesson 13).
"""

from __future__ import annotations

import torch
from torch import Tensor


def to_batch(x: Tensor) -> Tensor:
    """Turn one example into a batch of one.

    (d,) -> (1, d)
    """
    return x.unsqueeze(0)


def flatten_images(images: Tensor) -> Tensor:
    """Flatten each image into one long vector, keeping the batch dimension.

    (N, C, H, W) -> (N, C*H*W)
    """
    return images.flatten(1)


def channels_last(images: Tensor) -> Tensor:
    """Move the channel dimension to the end.

    (N, C, H, W) -> (N, H, W, C)
    """
    return images.permute(0, 2, 3, 1)


def normalize_columns(x: Tensor) -> Tensor:
    """Give every column mean 0 and standard deviation 1.

    (N, d) -> (N, d)
    Use the population standard deviation (``correction=0``).
    """
    mean = x.mean(dim=0)                  # (d,)
    std = x.std(dim=0, correction=0)      # (d,)
    return (x - mean) / std               # (N, d) - (d,) broadcasts over the rows


def pairwise_distance(a: Tensor, b: Tensor) -> Tensor:
    """Euclidean distance between every row of ``a`` and every row of ``b``.

    (N, d), (M, d) -> (N, M)
    """
    diff = a[:, None, :] - b[None, :, :]  # (N, 1, d) - (1, M, d) -> (N, M, d)
    return diff.pow(2).sum(dim=-1).sqrt()


def move_to(x: Tensor, device: str | torch.device) -> Tensor:
    """Return ``x`` living on ``device`` ('cpu', 'cuda', or a torch.device)."""
    return x.to(device)


def one_hot(indices: Tensor, num_classes: int) -> Tensor:
    """Stretch goal. Class ids become one-hot rows.

    (N,) int64 -> (N, num_classes) float32, exactly one 1.0 per row.
    """
    classes = torch.arange(num_classes, device=indices.device)   # (num_classes,)
    return (indices[:, None] == classes[None, :]).float()        # (N, 1) == (1, K) -> (N, K)
