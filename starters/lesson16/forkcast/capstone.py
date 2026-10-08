"""Lesson 16, the capstone. One model for a Forkcast listing: its photo, its description and its numbers, fused.

``make_listings``, ``ListingsDataset``, ``listing_loaders`` and ``accuracy_fusion`` are provided; the rest are exercises.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from forkcast.model import MLP
from forkcast.text import Vocab, pad_batch, tokenize
from forkcast.vision import SHAPE_CLASSES, make_shapes

MODALITIES = ("image", "text", "tabular")
TAB_FEATURES = ["price", "prep_minutes", "rating"]
DISH_PULL = [1.6, -0.5, 0.9, -1.5]            # how much each photo category helps demand (circle, square, triangle, cross)
PORTION = {"generous": 1.0, "decent": 0.3, "small": -0.8, "tiny": -1.2}
FRESH = {"fresh": 0.6, "homemade": 0.5, "standard": 0.0, "frozen": -0.7, "reheated": -0.9}


def make_listings(n: int = 1600, seed: int = 0, size: int = 32):
    """Synthetic listings. Returns (images (n, 3, size, size), texts: list[str], tabular (n, 3), labels (n,) int64).

    Demand depends on all three: the photo category (DISH_PULL), the words in the description (PORTION, FRESH),
    and price, prep time and rating. The label is 1 when demand is above the median: a balanced "will it sell" task
    that no single modality can solve on its own.
    """
    images, categories = make_shapes(n, seed=seed, size=size)
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed + 1)
    texts, text_score = [], []
    for _ in range(n):
        p, f = rng.choice(list(PORTION)), rng.choice(list(FRESH))
        name = rng.choice(["pad thai", "ramen", "pizza", "pho", "lasagna", "tacos", "curry", "burger"])
        texts.append(rng.choice([f"{p.capitalize()} portion of {f} {name}.", f"{f.capitalize()} {name}, {p} portion.", f"Our {name}: {f}, {p} serving."]))
        text_score.append(PORTION[p] + FRESH[f])
    price = 6 + 24 * torch.rand(n, generator=g)
    prep = 10 + 40 * torch.rand(n, generator=g)
    rating = 3.0 + 2.0 * torch.rand(n, generator=g)
    tabular = torch.stack([price, prep, rating], dim=1)
    demand = (torch.tensor(DISH_PULL)[categories] + 1.0 * torch.tensor(text_score) - 0.08 * price + 0.9 * rating - 0.02 * prep
              + 0.4 * torch.randn(n, generator=g))
    labels = (demand > demand.median()).long()
    return images, texts, tabular, labels


class ListingsDataset(Dataset):
    def __init__(self, images: Tensor, texts: list[str], tabular: Tensor, labels: Tensor, vocab: Vocab, tab_mean: Tensor, tab_std: Tensor):
        self.images, self.labels = images, labels
        self.ids = [torch.tensor(vocab.encode(tokenize(t)), dtype=torch.long) for t in texts]
        self.tabular = (tabular - tab_mean) / tab_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.images[i], self.ids[i], self.tabular[i], self.labels[i]


def collate_listings(batch):
    """A list of (image, ids, tabular, label) -> (images (N, 3, H, W), ids (N, T), mask (N, T), tabular (N, 3), labels (N,)).

    Stack what stacks, pad the text with pad_batch, and keep the label last for the toolkit.
    """
    raise NotImplementedError("Lesson 16, step 2: images, ids, tabular, labels = zip(*batch); pad_batch(list(ids)); torch.stack the rest")


class FusionModel(nn.Module):
    """Three encoders, one head. Each encoder maps its modality to a d-vector; the three are concatenated and classified.

    image: SmallCNN's two conv blocks (3 -> 16 -> 32 channels, two pools) then Flatten and Linear(32 * (size//4)^2, d).
    text: nn.Embedding(vocab, d, padding_idx=0) with Lesson 11's masked mean.
    tabular: MLP(3, (d,)) from Lesson 4 but with d outputs: use nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d)).
    head: nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Linear(hidden, 2)).
    ``modalities`` names the inputs that are used; an unused modality contributes a zero vector, so the same
    class runs every ablation.
    """

    def __init__(self, vocab_size: int, d: int = 32, hidden: int = 64, size: int = 32, modalities=MODALITIES):
        super().__init__()
        self.modalities = tuple(modalities)
        self.d = d
        raise NotImplementedError("Lesson 16, step 3: self.image_encoder, self.image_proj, self.text_embedding, self.tab_encoder, self.head")

    def encode_image(self, images: Tensor) -> Tensor:
        raise NotImplementedError("Lesson 16, step 3: conv blocks -> flatten -> Linear to d")

    def encode_text(self, ids: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError("Lesson 16, step 3: embedding, masked mean (Lesson 11)")

    def forward(self, images: Tensor, ids: Tensor, mask: Tensor, tabular: Tensor) -> Tensor:
        """Concatenate the used encoders' outputs (zeros for the unused ones) and classify. -> (N, 2)"""
        raise NotImplementedError("Lesson 16, step 3: encode each used modality, zeros of shape (N, d) for unused ones, torch.cat(dim=1), head")


def listing_loaders(n_train: int = 1200, n_val: int = 400, batch_size: int = 64):
    """Train and validation loaders plus the vocabulary and the tabular statistics (from the training part only)."""
    images, texts, tabular, labels = make_listings(n_train + n_val, seed=0)
    vocab = Vocab([tokenize(t) for t in texts[:n_train]])
    tab_mean, tab_std = tabular[:n_train].mean(dim=0), tabular[:n_train].std(dim=0, correction=0)
    train_ds = ListingsDataset(images[:n_train], texts[:n_train], tabular[:n_train], labels[:n_train], vocab, tab_mean, tab_std)
    val_ds = ListingsDataset(images[n_train:], texts[n_train:], tabular[n_train:], labels[n_train:], vocab, tab_mean, tab_std)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_listings),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_listings), vocab, (tab_mean, tab_std))


@torch.no_grad()
def accuracy_fusion(model: nn.Module, loader: DataLoader, device="cpu") -> float:
    model.eval(); correct = total = 0
    for *inputs, y in loader:
        inputs = [x.to(device) for x in inputs]
        correct += (model(*inputs).argmax(dim=1).cpu() == y).sum().item(); total += len(y)
    return correct / total


def run_ablation(make_model, train_loader: DataLoader, val_loader: DataLoader, combos, epochs: int = 8, lr: float = 1e-3, device="cpu") -> dict:
    """For each tuple of modalities in ``combos``: build make_model(modalities), train it with the toolkit's
    fit_v2 (CrossEntropyLoss), and record validation accuracy. Returns {combo: accuracy}. Seed with torch.manual_seed(0)
    before building each model so the comparison is fair."""
    raise NotImplementedError("Lesson 16, step 4: loop over combos, torch.manual_seed(0), train.fit_v2(..., loss_fn=nn.CrossEntropyLoss()), accuracy_fusion")


def write_model_card(path: str | Path, name: str, intended_use: str, data: str, metrics: dict, limitations: list[str], lesson_trail: list[str]) -> str:
    """A Markdown model card with the sections every deployed model should ship with. Returns the text and writes it.

    Sections, as level-2 headings in this order: Intended use, Training data, Evaluation, Limitations, Provenance.
    Evaluation renders ``metrics`` as a two-column table (metric | value, values formatted with 3 decimals when numeric).
    Limitations renders one bullet per entry; Provenance lists the lesson trail (which lesson built each part) as bullets.
    """
    raise NotImplementedError("Lesson 16, step 6: build the Markdown string section by section, write it, return it")
