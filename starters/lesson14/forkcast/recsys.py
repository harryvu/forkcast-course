"""Lesson 14. Recommendations: learn what a customer likes from what they ordered.

``make_order_history``, ``leave_one_out``, ``seen_items`` and ``popularity_hit_rate`` are provided; the rest are exercises.
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

N_USERS, N_RESTAURANTS = 300, 120


def make_order_history(n_users: int = N_USERS, n_items: int = N_RESTAURANTS, orders_per_user: int = 20, seed: int = 0):
    """Synthetic orders driven by hidden tastes, so the course can check what the model recovered.

    Every user and every restaurant gets a hidden 4-d taste vector; a user orders from ``orders_per_user``
    distinct restaurants drawn in proportion to softmax(2 * user . restaurant + popularity). Returns
    (frame with columns user, item, in time order per user; the hidden user factors; the hidden item factors).
    """
    g = torch.Generator().manual_seed(seed)
    user_f = torch.randn(n_users, 4, generator=g)
    item_f = torch.randn(n_items, 4, generator=g)
    popularity = 0.5 * torch.randn(n_items, generator=g)
    rows = []
    for u in range(n_users):
        probs = torch.softmax(2.0 * (item_f @ user_f[u]) + popularity, dim=0)
        items = torch.multinomial(probs, orders_per_user, replacement=False, generator=g)   # distinct restaurants
        rows += [(u, int(i)) for i in items.tolist()]
    frame = pd.DataFrame(rows, columns=["user", "item"])
    return frame, user_f, item_f


def leave_one_out(frame: pd.DataFrame):
    """The last order of every user becomes the test pair; everything before it is training (Lesson 5's split, per user)."""
    last = frame.groupby("user").tail(1)
    train = frame.drop(last.index).reset_index(drop=True)
    return train, last.reset_index(drop=True)


def seen_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    """user -> the set of restaurants in the frame. Used to avoid recommending what they already ordered."""
    return {int(u): set(int(i) for i in g["item"]) for u, g in frame.groupby("user")}


class MatrixFactorization(nn.Module):
    """score(user, item) = user_vec . item_vec + user_bias + item_bias + global_bias.

    Two nn.Embedding tables of dim ``dim`` (Lesson 6's zones, Lesson 11's words: the same layer again),
    two nn.Embedding(…, 1) tables for the biases, and one learnable scalar (nn.Parameter(torch.zeros(1))).
    Initialise the vector tables with std 0.05 so early scores are small.
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 16):
        super().__init__()
        raise NotImplementedError("Lesson 14, step 3: user_vec, item_vec (dim), user_bias, item_bias (1), global_bias; nn.init.normal_(…, std=0.05) on the vectors")

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        """(N,), (N,) int64 -> (N,) scores (logits for "this user orders here")."""
        raise NotImplementedError("Lesson 14, step 3: (self.user_vec(users) * self.item_vec(items)).sum(1) + biases")

    def score_all(self, users: Tensor) -> Tensor:
        """(N,) users -> (N, n_items): the score of every restaurant for each user, for ranking."""
        raise NotImplementedError("Lesson 14, step 3: self.user_vec(users) @ self.item_vec.weight.T + the broadcast biases")


def sample_negatives(users: Tensor, n_items: int, seen: dict[int, set[int]], generator: torch.Generator | None = None) -> Tensor:
    """One restaurant the user has NOT ordered from, per row of ``users``. (N,) -> (N,) int64.

    Draw uniformly with torch.randint; redraw any that the user has seen (a handful of loop passes).
    Implicit feedback has no "disliked" label: we treat unordered restaurants as weak negatives.
    """
    raise NotImplementedError("Lesson 14, step 4: randint, then redraw collisions with the seen sets")


def train_one_epoch_rec(model: MatrixFactorization, loader: DataLoader, optimizer: torch.optim.Optimizer, n_items: int, seen: dict[int, set[int]]) -> float:
    """One pass over the (user, item) positives. For each batch, sample one negative per positive and minimise
    BCEWithLogits(score(user, positive), 1) + BCEWithLogits(score(user, negative), 0). Returns the mean loss."""
    raise NotImplementedError("Lesson 14, step 4: zero_grad, positives and sampled negatives, two BCE terms, backward, step")


@torch.no_grad()
def hit_rate_at_k(model: MatrixFactorization, test: pd.DataFrame, seen: dict[int, set[int]], k: int = 10) -> float:
    """For each test (user, item): rank all restaurants by score, hide the ones the user already ordered from
    (set their score to -inf), and count a hit if the held-out item is in the top k. Returns hits / users."""
    raise NotImplementedError("Lesson 14, step 5: score_all, mask seen with -inf, topk, compare")


@torch.no_grad()
def nearest_restaurants(model: MatrixFactorization, item: int, k: int = 5) -> list[int]:
    """The k restaurants whose learned vectors are closest to ``item``'s by cosine similarity (excluding itself)."""
    raise NotImplementedError("Lesson 14, step 6: F.normalize the item table, cosine with the chosen row, topk(k + 1), drop itself")


def popularity_hit_rate(train: pd.DataFrame, test: pd.DataFrame, seen: dict[int, set[int]], k: int = 10) -> float:
    """The baseline every recommender must beat: recommend the most-ordered restaurants the user has not tried."""
    counts = train["item"].value_counts()
    ranked = [int(i) for i in counts.index]
    hits = 0
    for u, item in zip(test["user"], test["item"]):
        recs = [i for i in ranked if i not in seen[int(u)]][:k]
        hits += int(int(item) in recs)
    return hits / len(test)


def make_rec_loader(train: pd.DataFrame, batch_size: int = 256) -> DataLoader:
    users = torch.tensor(train["user"].values, dtype=torch.long)
    items = torch.tensor(train["item"].values, dtype=torch.long)
    return DataLoader(TensorDataset(users, items), batch_size=batch_size, shuffle=True)
