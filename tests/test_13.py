"""Lesson 13 definition of done.

Run:  pytest tests/test_13.py -q
Expected: 7 passed.   (A toy DistilBERT and a tokenizer trained on the bundled reviews: nothing is downloaded.)
"""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from conftest import lesson_module

pytest.importorskip("transformers", reason="pip install transformers (present in Colab)")
lm = lesson_module("lm", "encode_batch", "ClassificationCollator", "mean_pool", "embed_texts", "search", "clip_and_step", "fit_lm", lesson="Lesson 13")
text = lesson_module("text", "load_reviews", lesson="Lesson 11")

fast = pytest.mark.fast


@pytest.fixture(scope="module")
def setup():
    frame = text.load_reviews()
    tok = lm.train_wordpiece_tokenizer(frame["text"], vocab_size=600)
    torch.manual_seed(0)
    return frame, tok


@fast
def test_tokenizer_and_encode_batch(setup):
    frame, tok = setup
    enc = lm.encode_batch(tok, ["Great pizza, fast delivery!", "Cold."], max_length=32)
    assert set(enc) >= {"input_ids", "attention_mask"}
    assert enc["input_ids"].shape == enc["attention_mask"].shape and enc["input_ids"].shape[0] == 2
    assert enc["input_ids"][0, 0].item() == tok.cls_token_id and enc["attention_mask"].dtype == torch.long
    short = enc["attention_mask"][1]
    assert short.sum() < len(short) and (enc["input_ids"][1][short == 0] == tok.pad_token_id).all(), "the short text is padded with the pad id"
    pieces = tok.convert_ids_to_tokens(enc["input_ids"][0][enc["attention_mask"][0] == 1].tolist())
    assert pieces[0] == "[CLS]" and pieces[-1] == "[SEP]" and "pizza" in "".join(pieces).replace("##", "")


@fast
def test_collator_adds_labels(setup):
    frame, tok = setup
    collate = lm.ClassificationCollator(tok, max_length=32)
    loader = DataLoader(list(zip(frame["text"][:16], frame["label"][:16])), batch_size=8, collate_fn=collate)
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[0] == 8 and batch["labels"].shape == (8,) and batch["labels"].dtype == torch.long


@fast
def test_mean_pool_ignores_padding():
    torch.manual_seed(0)
    hidden = torch.randn(1, 5, 8)
    full = lm.mean_pool(hidden, torch.ones(1, 5, dtype=torch.long))
    assert torch.allclose(full, hidden.mean(dim=1))
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    assert torch.allclose(lm.mean_pool(hidden, mask), hidden[:, :3].mean(dim=1), atol=1e-6)


@fast
def test_embed_texts_gives_unit_vectors(setup):
    frame, tok = setup
    torch.manual_seed(0)
    enc = lm.tiny_encoder(tok.vocab_size)
    emb = lm.embed_texts(enc, tok, frame["text"][:10], batch_size=4)
    assert emb.shape == (10, 32) and not emb.requires_grad
    assert torch.allclose(emb.norm(dim=1), torch.ones(10), atol=1e-5), "normalise so that dot products are cosines"


@fast
def test_search_finds_the_near_duplicate(setup):
    frame, tok = setup
    torch.manual_seed(0)
    enc = lm.tiny_encoder(tok.vocab_size)
    menu = ["spicy chicken ramen with soft egg", "margherita pizza with fresh basil", "beef pho with lime and herbs", "green curry with jasmine rice", "crispy fish tacos"]
    emb = lm.embed_texts(enc, tok, menu)
    hits = lm.search("spicy chicken ramen with egg", menu, emb, enc, tok, k=2)
    assert hits[0][0] == menu[0] and -1.0 <= hits[0][1] <= 1.0001, f"the near-duplicate should rank first: {hits}"
    assert len(hits) == 2 and hits[0][1] >= hits[1][1]


@fast
def test_clip_and_step_bounds_the_gradient_norm():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    model.weight.grad = torch.full_like(model.weight, 10.0); model.bias.grad = torch.full_like(model.bias, 10.0)
    before = math.sqrt(5 * 100.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    returned = lm.clip_and_step(model, opt, max_grad_norm=1.0)
    assert abs(returned - before) < 1e-4, "return the norm found before clipping"
    assert model.weight.grad is None and model.bias.grad is None, "zero the gradients (set_to_none) after the step"


@fast
def test_fit_lm_accumulates_and_learns(setup):
    frame, tok = setup
    collate = lm.ClassificationCollator(tok, max_length=32)
    cut = 640
    train_loader = DataLoader(list(zip(frame["text"][:cut], frame["label"][:cut])), batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(list(zip(frame["text"][cut:], frame["label"][cut:])), batch_size=64, collate_fn=collate)
    torch.manual_seed(0)
    model = lm.tiny_distilbert(tok.vocab_size, num_labels=2)
    history = lm.fit_lm(model, train_loader, val_loader, epochs=3, lr=1e-3, accumulation_steps=2)
    assert set(history) == {"train", "val", "val_acc", "steps"} and len(history["val_acc"]) == 3
    assert history["steps"] == 3 * math.ceil(len(train_loader) / 2), "one optimizer step per two batches"
    assert history["val_acc"][-1] > 0.85, f"a tiny transformer fine-tunes to the templated reviews: {history['val_acc'][-1]:.2f}"
