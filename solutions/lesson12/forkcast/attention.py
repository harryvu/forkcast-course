"""Lesson 12. Attention from scratch to a Transformer encoder.

``make_negation_reviews`` and ``negation_loaders`` are provided; the rest are exercises.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from forkcast.text import Vocab, make_text_loaders, tokenize

DISHES = ["pad thai", "pizza", "ramen", "pho", "lasagna", "sushi", "bibimbap", "tacos", "curry", "burger"]
# Each positive template has a negative twin made of exactly the same words in a different order.
PAIRS = [
    ("the {d} was good , not bad .", "the {d} was bad , not good ."),
    ("{d} : not bad , actually good !", "{d} : not good , actually bad !"),
    ("good {d} , not a bad delivery", "bad {d} , not a good delivery"),
    ("great {d} and not terrible service", "terrible {d} and not great service"),
    ("the rider was great , the {d} not terrible", "the rider was terrible , the {d} not great"),
]


def make_negation_reviews(n: int = 600, seed: int = 0) -> tuple[list[str], list[int]]:
    """Reviews where word order decides the label: every positive review has a negative twin with the same words."""
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n // 2):
        pos, neg = rng.choice(PAIRS)
        d = rng.choice(DISHES)
        texts += [pos.format(d=d), neg.format(d=d)]
        labels += [1, 0]
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], [labels[i] for i in order]


def negation_loaders(n_train: int = 600, n_val: int = 200, batch_size: int = 32):
    """Loaders plus the vocabulary, reusing Lesson 11's text pipeline."""
    tr_x, tr_y = make_negation_reviews(n_train, seed=0)
    va_x, va_y = make_negation_reviews(n_val, seed=1)
    vocab = Vocab([tokenize(t) for t in tr_x])
    train_loader, val_loader = make_text_loaders(tr_x, tr_y, va_x, va_y, vocab, batch_size=batch_size)
    return train_loader, val_loader, vocab


def scaled_dot_product(q: Tensor, k: Tensor, v: Tensor, key_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Attention by hand. q, k, v: (N, T, d). key_mask: (N, T) bool, True on real tokens.

    scores = q @ k^T / sqrt(d)                      (N, T, T): row i says how much token i looks at each token j
    padded keys get -inf before the softmax         (so their weight is exactly 0)
    weights = softmax over the last dimension        each row sums to 1
    output = weights @ v                             (N, T, d)
    Return (output, weights).
    """
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)               # (N, T, T)
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, :], float("-inf"))   # a padded key is never looked at
    weights = torch.softmax(scores, dim=-1)                       # each row sums to 1
    return weights @ v, weights                                   # (N, T, d), (N, T, T)


class PositionalEncoding(nn.Module):
    """Adds a fixed sin/cos pattern per position so the model can tell "not good" from "good not".

    Build a table pe of shape (1, max_len, d_model): pe[0, pos, 2i] = sin(pos / 10000^(2i/d)),
    pe[0, pos, 2i+1] = cos(pos / 10000^(2i/d)). Store it with ``self.register_buffer("pe", pe)``:
    part of the model's state, moved with .to(device), saved in the state_dict, but not a parameter.
    forward(x) returns x + pe[:, :x.shape[1]].
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)                                    # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))     # (d_model/2,)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)                       # state, not a parameter: no gradient, but saved and moved with the model

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.shape[1]]


class TinyTransformer(nn.Module):
    """Embedding + positions -> a stack of encoder layers -> masked mean -> classifier.

    Use nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True) wrapped in
    nn.TransformerEncoder(layer, num_layers). The encoder's ``src_key_padding_mask`` is True where tokens
    must be IGNORED, so pass ``~mask``. Pool with Lesson 11's masked mean, then a Linear head.
    """

    def __init__(self, vocab_size: int, d_model: int = 32, nhead: int = 2, num_layers: int = 1, dim_feedforward: int = 64, dropout: float = 0.1, n_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positions = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)   # plain path: no prototype warnings, same result
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        x = self.positions(self.embedding(ids))                              # (N, T, d_model)
        x = self.encoder(x, src_key_padding_mask=~mask)                      # True = ignore this key
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)            # (N, d_model)
        return self.head(pooled)
