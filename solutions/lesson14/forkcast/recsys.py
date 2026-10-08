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
        self.user_vec = nn.Embedding(n_users, dim)
        self.item_vec = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        for table in (self.user_vec, self.item_vec):
            nn.init.normal_(table.weight, std=0.05)
        for table in (self.user_bias, self.item_bias):
            nn.init.zeros_(table.weight)

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        """(N,), (N,) int64 -> (N,) scores (logits for "this user orders here")."""
        dot = (self.user_vec(users) * self.item_vec(items)).sum(dim=1)                    # (N,)
        return dot + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias

    def score_all(self, users: Tensor) -> Tensor:
        """(N,) users -> (N, n_items): the score of every restaurant for each user, for ranking."""
        scores = self.user_vec(users) @ self.item_vec.weight.T                               # (N, n_items)
        return scores + self.user_bias(users) + self.item_bias.weight.T + self.global_bias     # (N, 1) + (1, n_items) broadcast


def sample_negatives(users: Tensor, n_items: int, seen: dict[int, set[int]], generator: torch.Generator | None = None) -> Tensor:
    """One restaurant the user has NOT ordered from, per row of ``users``. (N,) -> (N,) int64.

    Draw uniformly with torch.randint; redraw any that the user has seen (a handful of loop passes).
    Implicit feedback has no "disliked" label: we treat unordered restaurants as weak negatives.
    """
    negatives = torch.randint(0, n_items, users.shape, generator=generator)
    for _ in range(10):                                                           # redraw collisions; a few passes is plenty
        clash = torch.tensor([int(n) in seen.get(int(u), set()) for u, n in zip(users.tolist(), negatives.tolist())])
        if not clash.any():
            break
        negatives[clash] = torch.randint(0, n_items, (int(clash.sum()),), generator=generator)
    return negatives


def train_one_epoch_rec(model: MatrixFactorization, loader: DataLoader, optimizer: torch.optim.Optimizer, n_items: int, seen: dict[int, set[int]]) -> float:
    """One pass over the (user, item) positives. For each batch, sample one negative per positive and minimise
    BCEWithLogits(score(user, positive), 1) + BCEWithLogits(score(user, negative), 0). Returns the mean loss."""
    model.train()
    total, seen_rows = 0.0, 0
    for users, positives in loader:
        negatives = sample_negatives(users, n_items, seen)
        optimizer.zero_grad()
        pos_scores = model(users, positives)
        neg_scores = model(users, negatives)
        loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
               F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss.backward()
        optimizer.step()
        total += loss.item() * len(users); seen_rows += len(users)
    return total / seen_rows


@torch.no_grad()
def hit_rate_at_k(model: MatrixFactorization, test: pd.DataFrame, seen: dict[int, set[int]], k: int = 10) -> float:
    """For each test (user, item): rank all restaurants by score, hide the ones the user already ordered from
    (set their score to -inf), and count a hit if the held-out item is in the top k. Returns hits / users."""
    model.eval()
    users = torch.tensor(test["user"].values, dtype=torch.long)
    targets = torch.tensor(test["item"].values, dtype=torch.long)
    scores = model.score_all(users)                                                       # (N, n_items)
    for row, u in enumerate(users.tolist()):
        for i in seen.get(u, ()):
            scores[row, i] = float("-inf")                                                # never recommend what they already ordered
    top = scores.topk(k, dim=1).indices                                                   # (N, k)
    return (top == targets[:, None]).any(dim=1).float().mean().item()


@torch.no_grad()
def nearest_restaurants(model: MatrixFactorization, item: int, k: int = 5) -> list[int]:
    """The k restaurants whose learned vectors are closest to ``item``'s by cosine similarity (excluding itself)."""
    table = F.normalize(model.item_vec.weight, dim=1)
    sims = table @ table[item]
    return [int(i) for i in sims.topk(k + 1).indices.tolist() if i != item][:k]


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
