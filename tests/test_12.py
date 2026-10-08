"""Lesson 12 definition of done.

Run:  pytest tests/test_12.py -q
Expected: 7 passed.
"""

from collections import Counter

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from conftest import lesson_module

attention = lesson_module("attention", "scaled_dot_product", "PositionalEncoding", "TinyTransformer", lesson="Lesson 12")
text = lesson_module("text", "BagOfEmbeddings", lesson="Lesson 11")
train = lesson_module("train", "fit_v2", lesson="Lesson 8")

fast = pytest.mark.fast


def accuracy(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for ids, mask, y in loader:
            correct += (model(ids, mask).argmax(1) == y).sum().item(); total += len(y)
    return correct / total


@fast
def test_negation_reviews_have_identical_bags_of_words():
    texts, labels = attention.make_negation_reviews(200, seed=0)
    assert len(texts) == 200 and sum(labels) == 100
    pos = Counter(frozenset(Counter(t.split()).items()) for t, l in zip(texts, labels) if l == 1)
    neg = Counter(frozenset(Counter(t.split()).items()) for t, l in zip(texts, labels) if l == 0)
    assert pos == neg, "every positive review's bag of words must appear as a negative review too"


@fast
def test_scaled_dot_product_matches_pytorch():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    out, w = attention.scaled_dot_product(q, k, v)
    assert out.shape == (2, 5, 8) and w.shape == (2, 5, 5)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 5), atol=1e-6), "each row of attention weights sums to 1"
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5), "must match PyTorch's fused version"


@fast
def test_padded_keys_get_zero_weight():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    key_mask = torch.tensor([[True, True, True, False]])
    out, w = attention.scaled_dot_product(q, k, v, key_mask)
    assert torch.all(w[..., 3] == 0), "the padded key must receive exactly zero attention"
    out_ref, _ = attention.scaled_dot_product(q, k[:, :3], v[:, :3])
    assert torch.allclose(out, out_ref, atol=1e-6), "masking a key is the same as removing it"


@fast
def test_positional_encoding_is_a_buffer_with_distinct_positions():
    pe = attention.PositionalEncoding(d_model=16, max_len=50)
    assert pe.pe.shape == (1, 50, 16)
    assert sum(p.numel() for p in pe.parameters()) == 0, "the table is a buffer, not a parameter"
    assert "pe" in pe.state_dict(), "buffers are saved with the model"
    assert not torch.allclose(pe.pe[0, 0], pe.pe[0, 1]) and not torch.allclose(pe.pe[0, 3], pe.pe[0, 17])
    x = torch.zeros(2, 7, 16)
    out = pe(x)
    assert out.shape == (2, 7, 16) and torch.allclose(out[0], out[1]), "the same position gets the same code in every sequence"
    assert torch.allclose(pe.pe[0, 0, 0::2], torch.zeros(8)) and torch.allclose(pe.pe[0, 0, 1::2], torch.ones(8)), "position 0 is sin(0)=0 and cos(0)=1"


@fast
def test_tiny_transformer_sees_order_and_ignores_padding():
    torch.manual_seed(0)
    model = attention.TinyTransformer(vocab_size=50, d_model=16, nhead=2, dropout=0.0).eval()
    ids = torch.tensor([[5, 6, 7, 8]]); swapped = torch.tensor([[5, 8, 7, 6]])
    assert model(ids, ids != 0).shape == (1, 2)
    assert not torch.allclose(model(ids, ids != 0), model(swapped, swapped != 0)), "swapping two tokens must change the output: the model sees order"
    padded = torch.tensor([[5, 6, 7, 8, 0, 0]])
    assert torch.allclose(model(ids, ids != 0), model(padded, padded != 0), atol=1e-5), "padding must change nothing"


@fast
def test_bag_of_embeddings_cannot_learn_negation():
    train_loader, val_loader, vocab = attention.negation_loaders()
    torch.manual_seed(0)
    bag = text.BagOfEmbeddings(len(vocab), dim=32, hidden=32)
    train.fit_v2(bag, train_loader, val_loader, epochs=15, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
    acc = accuracy(bag, val_loader)
    assert acc < 0.65, f"same words in both classes: a bag should be stuck near 50%, got {acc:.2f}"


@fast
def test_tiny_transformer_learns_negation():
    train_loader, val_loader, vocab = attention.negation_loaders()
    torch.manual_seed(0)
    model = attention.TinyTransformer(len(vocab), d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.1)
    train.fit_v2(model, train_loader, val_loader, epochs=25, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
    acc = accuracy(model, val_loader)
    assert acc > 0.9, f"with positions and attention the order is learnable: got {acc:.2f}"
