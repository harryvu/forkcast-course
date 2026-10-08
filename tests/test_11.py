"""Lesson 11 definition of done.

Run:  pytest tests/test_11.py -q
Expected: 7 passed.
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module

text = lesson_module("text", "tokenize", "Vocab", "pad_batch", "collate_reviews", "BagOfEmbeddings", lesson="Lesson 11")
train = lesson_module("train", "fit_v2", lesson="Lesson 8")

fast = pytest.mark.fast


@pytest.fixture(scope="module")
def corpus():
    frame = text.load_reviews()
    cut = int(0.8 * len(frame))
    tr, va = frame.iloc[:cut], frame.iloc[cut:]
    vocab = text.Vocab([text.tokenize(t) for t in tr["text"]], min_freq=1)
    return tr, va, vocab


@fast
def test_tokenize_lowercases_and_splits_punctuation():
    assert text.tokenize("Great food, FAST delivery!") == ["great", "food", ",", "fast", "delivery", "!"]
    assert text.tokenize("Didn't arrive... Never again?") == ["didn't", "arrive", ".", ".", ".", "never", "again", "?"]


@fast
def test_vocab_specials_frequency_order_and_unknowns():
    v = text.Vocab([["hot", "hot", "hot", "cold", "cold", "late"]], min_freq=1)
    assert v.itos[:2] == ["<pad>", "<unk>"]
    assert v.itos[2:] == ["hot", "cold", "late"], "most frequent tokens first"
    assert v.encode(["hot", "soggy", "late"]) == [2, 1, 4], "unknown tokens map to 1"
    assert v.decode([2, 1, 0]) == ["hot", "<unk>", "<pad>"]
    assert len(text.Vocab([["a", "a", "b"]], min_freq=2)) == 3, "min_freq drops rare tokens"
    assert len(text.Vocab([["a", "b", "c", "d"]], max_size=4)) == 4, "max_size counts the two specials"


@fast
def test_pad_batch_shapes_and_mask():
    seqs = [torch.tensor([5, 6, 7]), torch.tensor([8, 9, 10, 11, 12]), torch.tensor([13, 14])]
    ids, mask = text.pad_batch(seqs)
    assert ids.shape == (3, 5) and mask.shape == (3, 5) and ids.dtype == torch.long and mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [3, 5, 2], "the mask counts the real tokens"
    assert torch.equal(ids[0], torch.tensor([5, 6, 7, 0, 0])), "padding uses index 0"
    assert (ids[~mask] == 0).all()


@fast
def test_collate_gives_ids_mask_then_labels(corpus):
    tr, va, vocab = corpus
    train_loader, val_loader = text.make_text_loaders(tr["text"], tr["label"], va["text"], va["label"], vocab, batch_size=16)
    ids, mask, labels = next(iter(train_loader))
    assert ids.shape[0] == 16 and ids.shape == mask.shape and labels.shape == (16,)
    assert ids.dtype == torch.long and mask.dtype == torch.bool and labels.dtype == torch.long
    assert mask.any(dim=1).all(), "every review has at least one real token"


@fast
def test_bag_of_embeddings_ignores_padding(corpus):
    _, _, vocab = corpus
    torch.manual_seed(0)
    model = text.BagOfEmbeddings(len(vocab), dim=16, hidden=16)
    short = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0, 0]])
    out_short = model(short, short != 0)
    out_padded = model(padded, padded != 0)
    assert out_short.shape == (1, 2)
    assert torch.allclose(out_short, out_padded, atol=1e-6), "padding must not change the prediction: that is what the mask is for"
    assert torch.equal(model.embedding.weight[0], torch.zeros(16)), "padding_idx=0 keeps row 0 at zero"


@fast
def test_padding_row_gets_no_gradient(corpus):
    _, _, vocab = corpus
    torch.manual_seed(0)
    model = text.BagOfEmbeddings(len(vocab), dim=16, hidden=16)
    ids = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
    nn.CrossEntropyLoss()(model(ids, ids != 0), torch.tensor([1, 0])).backward()
    grad = model.embedding.weight.grad
    assert grad is not None and torch.equal(grad[0], torch.zeros(16)), "the <pad> row must receive zero gradient"
    assert grad[5].abs().sum() > 0, "real tokens do learn"


@fast
def test_classifier_learns_the_reviews(corpus):
    tr, va, vocab = corpus
    train_loader, val_loader = text.make_text_loaders(tr["text"], tr["label"], va["text"], va["label"], vocab, batch_size=32)
    torch.manual_seed(0)
    model = text.BagOfEmbeddings(len(vocab), dim=32, hidden=32)
    train.fit_v2(model, train_loader, val_loader, epochs=8, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for ids, mask, labels in val_loader:
            correct += (model(ids, mask).argmax(1) == labels).sum().item(); total += len(labels)
    assert correct / total > 0.9, f"templated reviews are easy: accuracy {correct / total:.2f}"
