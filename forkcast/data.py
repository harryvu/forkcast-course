"""Lesson 1. Synthetic Forkcast deliveries.

The data is made up on purpose: because we know the rule that produced it,
we can check what a model learned against the truth. Real data arrives in
Lesson 5. Lesson 2 adds more features to this generator.
"""

from __future__ import annotations

import torch
from torch import Tensor

TRUE_INTERCEPT = 8.0   # minutes: the kitchen's preparation time
TRUE_SLOPE = 3.5       # minutes per km of riding


def make_deliveries(n: int = 200, seed: int = 0, noise: float = 2.0) -> tuple[Tensor, Tensor]:
    """Return ``(distance_km, minutes)``, two float32 tensors of shape (n,).

    minutes = TRUE_INTERCEPT + TRUE_SLOPE * distance_km + gaussian noise
    Distances are uniform between 0.5 and 10 km.
    """
    g = torch.Generator().manual_seed(seed)
    distance = 0.5 + 9.5 * torch.rand(n, generator=g)
    minutes = TRUE_INTERCEPT + TRUE_SLOPE * distance + noise * torch.randn(n, generator=g)
    return distance, minutes


# ---- Lesson 2: four features ------------------------------------------------

FEATURES = ["distance_km", "items", "rain", "rush_hour"]
TRUE_WEIGHTS = torch.tensor([3.5, 1.5, 4.0, 6.0])   # minutes per km, per item, if raining, at rush hour
TRUE_BIAS = 8.0                                     # the kitchen's preparation time


def make_orders(n: int = 400, seed: int = 0, noise: float = 2.0, nonlinear: bool = False) -> tuple[Tensor, Tensor]:
    """Return ``(X, minutes)``: features of shape (n, 4) and targets of shape (n,).

    Columns of X, in the order of ``FEATURES``: distance in km (0.5 to 10),
    number of items (1 to 6), rain (0 or 1), rush hour (0 or 1).

    Linear (Lessons 2 and 3):   minutes = TRUE_BIAS + X @ TRUE_WEIGHTS + noise
    Nonlinear (Lesson 4 on):    the same, plus two effects no straight line can express:
      rain slows every kilometre (+1.5 min per km when raining), and beyond 7 km
      riders must take the bike-lane detour (+5 min per km past 7 km).
    """
    g = torch.Generator().manual_seed(seed)
    distance = 0.5 + 9.5 * torch.rand(n, generator=g)
    items = torch.randint(1, 7, (n,), generator=g).float()
    rain = (torch.rand(n, generator=g) < 0.3).float()
    rush = (torch.rand(n, generator=g) < 0.25).float()
    X = torch.stack([distance, items, rain, rush], dim=1)          # (n, 4)
    minutes = TRUE_BIAS + X @ TRUE_WEIGHTS
    if nonlinear:
        minutes = minutes + 1.5 * rain * distance + 5.0 * torch.relu(distance - 7.0)
    minutes = minutes + noise * torch.randn(n, generator=g)
    return X, minutes
