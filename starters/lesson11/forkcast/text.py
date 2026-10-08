"""Lesson 11. Text as tensors: tokens, a vocabulary, padding and masks, and a bag-of-embeddings classifier.

``load_reviews`` and ``make_text_loaders`` are provided; the rest are exercises.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD, UNK = "<pad>", "<unk>"
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "reviews_mini.csv"
_TOKEN = re.compile(r"[a-z0-9']+|[!?.,]")


def tokenize(text: str) -> list[str]:
    """Lower-case the text and split it into words and the punctuation marks ! ? . ,

    "Great food, FAST delivery!" -> ["great", "food", ",", "fast", "delivery", "!"]
    Use ``_TOKEN.findall`` on the lower-cased text.
    """
    raise NotImplementedError("Lesson 11, step 2: _TOKEN.findall(text.lower())")


class Vocab:
    """Token <-> integer id. Index 0 is PAD, index 1 is UNK, then tokens by decreasing frequency.

    ``min_freq`` drops rare tokens (they become UNK); ``max_size`` caps the total size including the two specials.
    Build it from the training texts only.
    """

    def __init__(self, token_lists, min_freq: int = 1, max_size: int | None = None):
        raise NotImplementedError("Lesson 11, step 3: count tokens with Counter; keep those with count >= min_freq, most common first; self.itos = [PAD, UNK] + kept[:max_size - 2]; self.stoi = {token: i}")

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        """Tokens to ids; unknown tokens become the UNK id (1)."""
        raise NotImplementedError("Lesson 11, step 3: [self.stoi.get(t, 1) for t in tokens]")

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]


def pad_batch(sequences: list[Tensor], pad_idx: int = 0) -> tuple[Tensor, Tensor]:
    """A list of 1-d int64 tensors of different lengths -> (ids (N, T), mask (N, T) bool).

    T is the longest length in the batch. ``pad_sequence(..., batch_first=True, padding_value=pad_idx)``
    does the padding; the mask is True where there is a real token.
    """
    raise NotImplementedError("Lesson 11, step 4: ids = pad_sequence(...); mask = ids != pad_idx; return ids, mask")


class ReviewsDataset(Dataset):
    """One review per item: (ids (T,) int64, label () int64), truncated to ``max_len`` tokens."""

    def __init__(self, texts, labels, vocab: Vocab, max_len: int = 200):
        self.ids = [torch.tensor(vocab.encode(tokenize(t))[:max_len], dtype=torch.long) for t in texts]
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.ids[i], self.labels[i]


def collate_reviews(batch) -> tuple[Tensor, Tensor, Tensor]:
    """The DataLoader's collate_fn: a list of (ids, label) -> (ids (N, T), mask (N, T), labels (N,)).

    Reviews have different lengths, so the default stacking fails; pad them with ``pad_batch``.
    The toolkit expects the target last, so return ids and mask first.
    """
    raise NotImplementedError("Lesson 11, step 4: ids, labels = zip(*batch); padded, mask = pad_batch(list(ids)); return padded, mask, torch.stack(labels)")


class BagOfEmbeddings(nn.Module):
    """Embed every token, average the real ones, classify the average.

    (N, T) ids -> Embedding(vocab, dim, padding_idx=0) -> (N, T, dim) -> masked mean over T -> (N, dim) -> MLP -> (N, n_classes)
    The masked mean multiplies by ``mask[..., None]``, sums over T, and divides by ``mask.sum(1, keepdim=True).clamp_min(1)``.
    """

    def __init__(self, vocab_size: int, dim: int = 64, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        raise NotImplementedError("Lesson 11, step 5: self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0); self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))")

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError("Lesson 11, step 5: emb = self.embedding(ids); masked mean over dim 1; return self.head(pooled)")


def load_reviews(path: str | Path = SAMPLE) -> pd.DataFrame:
    """Columns: text, label (1 = positive, 0 = negative)."""
    return pd.read_csv(path)


def make_text_loaders(train_texts, train_labels, val_texts, val_labels, vocab: Vocab, batch_size: int = 64, max_len: int = 200) -> tuple[DataLoader, DataLoader]:
    """Loaders over ReviewsDatasets with collate_reviews; the vocabulary comes from the training texts."""
    train_ds = ReviewsDataset(train_texts, train_labels, vocab, max_len)
    val_ds = ReviewsDataset(val_texts, val_labels, vocab, max_len)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_reviews),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_reviews))
