"""Lesson 0. Six small tensor functions.

Each docstring states the exact input and output shapes the tests expect.
Replace every ``raise NotImplementedError`` with one to three lines of code.
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
    raise NotImplementedError("Lesson 0, step 6: add a dimension of size 1 at the front")


def flatten_images(images: Tensor) -> Tensor:
    """Flatten each image into one long vector, keeping the batch dimension.

    (N, C, H, W) -> (N, C*H*W)
    """
    raise NotImplementedError("Lesson 0, step 6: merge everything after dim 0")


def channels_last(images: Tensor) -> Tensor:
    """Move the channel dimension to the end.

    (N, C, H, W) -> (N, H, W, C)
    """
    raise NotImplementedError("Lesson 0, step 6: reorder the dimensions, batch stays first")


def normalize_columns(x: Tensor) -> Tensor:
    """Give every column mean 0 and standard deviation 1.

    (N, d) -> (N, d)
    Use the population standard deviation (``correction=0``).
    """
    raise NotImplementedError("Lesson 0, step 6: mean and std over dim 0, then broadcast")


def pairwise_distance(a: Tensor, b: Tensor) -> Tensor:
    """Euclidean distance between every row of ``a`` and every row of ``b``.

    (N, d), (M, d) -> (N, M)
    """
    raise NotImplementedError("Lesson 0, step 6: build an (N, M, d) difference tensor with broadcasting")


def move_to(x: Tensor, device: str | torch.device) -> Tensor:
    """Return ``x`` living on ``device`` ('cpu', 'cuda', or a torch.device)."""
    raise NotImplementedError("Lesson 0, step 6: one method call")


def one_hot(indices: Tensor, num_classes: int) -> Tensor:
    """Stretch goal. Class ids become one-hot rows.

    (N,) int64 -> (N, num_classes) float32, exactly one 1.0 per row.
    Do it with torch.arange, a comparison and broadcasting.
    """
    raise NotImplementedError("Lesson 0, stretch: compare indices[:, None] with torch.arange(num_classes)")
