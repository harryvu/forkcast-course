"""Lesson 14 definition of done.

Run:  pytest tests/test_14.py -q
Expected: 7 passed.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from conftest import lesson_module

rec = lesson_module("recsys", "MatrixFactorization", "sample_negatives", "train_one_epoch_rec", "hit_rate_at_k", "nearest_restaurants", lesson="Lesson 14")

fast = pytest.mark.fast


@pytest.fixture(scope="module")
def data():
    frame, user_f, item_f = rec.make_order_history(seed=0)
    train, test = rec.leave_one_out(frame)
    return frame, user_f, item_f, train, test, rec.seen_items(train)


@fast
def test_order_history_and_split(data):
    frame, user_f, item_f, train, test, seen = data
    assert len(frame) == rec.N_USERS * 20 and frame["user"].nunique() == rec.N_USERS and frame["item"].max() < rec.N_RESTAURANTS
    assert len(test) == rec.N_USERS and test["user"].is_unique, "exactly one held-out order per user"
    assert len(train) + len(test) == len(frame)
    f2, _, _ = rec.make_order_history(seed=0)
    assert f2.equals(frame)


@fast
def test_matrix_factorization_shapes_and_formula():
    torch.manual_seed(0)
    model = rec.MatrixFactorization(10, 7, dim=4)
    users, items = torch.tensor([0, 3, 9]), torch.tensor([1, 6, 2])
    out = model(users, items)
    assert out.shape == (3,)
    by_hand = (model.user_vec(users) * model.item_vec(items)).sum(1) + model.user_bias(users).squeeze(1) + model.item_bias(items).squeeze(1) + model.global_bias
    assert torch.allclose(out, by_hand, atol=1e-6)
    allscores = model.score_all(users)
    assert allscores.shape == (3, 7)
    assert torch.allclose(allscores[torch.arange(3), items], out, atol=1e-6), "score_all must agree with forward for every (user, item)"


@fast
def test_sample_negatives_avoids_seen_items():
    users = torch.tensor([0, 0, 1, 1, 1])
    seen = {0: set(range(0, 95)), 1: {3, 4}}
    neg = rec.sample_negatives(users, 100, seen, generator=torch.Generator().manual_seed(0))
    assert neg.shape == (5,) and neg.dtype == torch.long and neg.min() >= 0 and neg.max() < 100
    assert all(int(n) not in seen[int(u)] for u, n in zip(users, neg)), "a sampled negative must not be a restaurant the user ordered from"


class _Oracle(nn.Module):
    """Scores the held-out item highest for every user: hit rate must be 1."""
    def __init__(self, test):
        super().__init__(); self.targets = {int(u): int(i) for u, i in zip(test["user"], test["item"])}
    def score_all(self, users):
        s = torch.zeros(len(users), rec.N_RESTAURANTS)
        for r, u in enumerate(users.tolist()): s[r, self.targets[u]] = 10.0
        return s


@fast
def test_hit_rate_at_k_definition(data):
    frame, user_f, item_f, train, test, seen = data
    assert rec.hit_rate_at_k(_Oracle(test), test, seen, k=1) == 1.0
    torch.manual_seed(0)
    model = rec.MatrixFactorization(rec.N_USERS, rec.N_RESTAURANTS, dim=8)
    h5, h10, h20 = (rec.hit_rate_at_k(model, test, seen, k=k) for k in (5, 10, 20))
    assert 0.0 <= h5 <= h10 <= h20 <= 1.0, "a larger k can only add hits"


@fast
def test_training_beats_the_popularity_baseline(data):
    frame, user_f, item_f, train, test, seen = data
    torch.manual_seed(0)
    model = rec.MatrixFactorization(rec.N_USERS, rec.N_RESTAURANTS, dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    loader = rec.make_rec_loader(train)
    losses = [rec.train_one_epoch_rec(model, loader, opt, rec.N_RESTAURANTS, seen) for _ in range(12)]
    assert losses[-1] < 0.6 * losses[0], f"the loss should fall clearly: {losses[0]:.3f} -> {losses[-1]:.3f}"
    popularity = rec.popularity_hit_rate(train, test, seen, k=10)
    learned = rec.hit_rate_at_k(model, test, seen, k=10)
    assert learned > popularity + 0.1, f"hit@10: learned {learned:.3f} must clearly beat popularity {popularity:.3f}"


@fast
def test_learned_item_vectors_recover_the_planted_tastes(data):
    frame, user_f, item_f, train, test, seen = data
    torch.manual_seed(0)
    model = rec.MatrixFactorization(rec.N_USERS, rec.N_RESTAURANTS, dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    loader = rec.make_rec_loader(train)
    for _ in range(12):
        rec.train_one_epoch_rec(model, loader, opt, rec.N_RESTAURANTS, seen)
    learned = F.normalize(model.item_vec.weight.detach(), dim=1); truth = F.normalize(item_f, dim=1)
    corr = torch.corrcoef(torch.stack([(learned @ learned.T).flatten(), (truth @ truth.T).flatten()]))[0, 1]
    assert corr > 0.3, f"learned restaurant similarities should mirror the hidden tastes: correlation {corr:.2f}"


@fast
def test_nearest_restaurants_excludes_itself():
    torch.manual_seed(0)
    model = rec.MatrixFactorization(5, 12, dim=4)
    near = rec.nearest_restaurants(model, 7, k=3)
    assert len(near) == 3 and 7 not in near and all(0 <= i < 12 for i in near)
    with torch.no_grad():
        model.item_vec.weight[2] = model.item_vec.weight[7] * 3            # same direction, different length
    assert rec.nearest_restaurants(model, 7, k=1) == [2], "cosine similarity ignores length: a scaled copy is the nearest"
