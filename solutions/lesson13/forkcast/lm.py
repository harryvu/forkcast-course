"""Lesson 13. Pretrained language models: a Hugging Face model is an nn.Module, fine-tune it gently, embed sentences.

``train_wordpiece_tokenizer``, ``tiny_distilbert`` and ``tiny_encoder`` are provided so the tests run offline on a
small random model; the notebook uses the real pretrained ones. The rest are exercises.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def train_wordpiece_tokenizer(texts, vocab_size: int = 1000):
    """A small subword tokenizer trained on our own texts, wrapped in the Hugging Face tokenizer API.

    Real models ship their tokenizer (AutoTokenizer.from_pretrained); this one exists so the tests need no download.
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.NFD(), normalizers.StripAccents()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
    tok.train_from_iterator(list(texts), trainer)
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")


def _tiny_config(vocab_size: int, num_labels: int = 2):
    from transformers import DistilBertConfig
    return DistilBertConfig(vocab_size=vocab_size, dim=32, n_layers=1, n_heads=2, hidden_dim=64, max_position_embeddings=128,
                            pad_token_id=0, num_labels=num_labels, dropout=0.0, attention_dropout=0.0)


def tiny_distilbert(vocab_size: int, num_labels: int = 2) -> nn.Module:
    """DistilBERT's architecture at toy size with random weights: the same class the notebook fine-tunes for real."""
    from transformers import DistilBertForSequenceClassification
    return DistilBertForSequenceClassification(_tiny_config(vocab_size, num_labels))


def tiny_encoder(vocab_size: int) -> nn.Module:
    """The encoder alone (no classification head), for sentence embeddings."""
    from transformers import DistilBertModel
    return DistilBertModel(_tiny_config(vocab_size))


def encode_batch(tokenizer, texts, max_length: int = 64) -> dict:
    """Texts -> a dict of tensors the model accepts: input_ids and attention_mask, both (N, T).

    ``tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")``.
    """
    return tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")


class ClassificationCollator:
    """DataLoader collate_fn for (text, label) pairs: tokenizes the batch and adds a ``labels`` tensor.

    Hugging Face classification models compute their own loss when ``labels`` is in the batch.
    """

    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, batch) -> dict:
        texts, labels = zip(*batch)
        enc = encode_batch(self.tokenizer, texts, self.max_length)
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return dict(enc)


def mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """(N, T, D), (N, T) -> (N, D): the mean over the real tokens. Lesson 11's masked mean, once more."""
    m = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)          # (N, T, 1)
    return (last_hidden_state * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def embed_texts(encoder: nn.Module, tokenizer, texts, device: str | torch.device = "cpu", batch_size: int = 32, max_length: int = 64) -> Tensor:
    """Sentence embeddings: mean-pooled encoder outputs, L2-normalised so dot products are cosine similarities. (N, D)

    encoder.eval(); for each batch: encode_batch -> move to device -> encoder(**batch).last_hidden_state -> mean_pool -> F.normalize.
    """
    encoder.eval()
    device = torch.device(device)
    out = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = {k: v.to(device) for k, v in encode_batch(tokenizer, texts[start:start + batch_size], max_length).items()}
        hidden = encoder(**batch).last_hidden_state                         # (n, T, D)
        out.append(F.normalize(mean_pool(hidden, batch["attention_mask"]), dim=-1).cpu())
    return torch.cat(out)


def search(query: str, texts, embeddings: Tensor, encoder: nn.Module, tokenizer, k: int = 3, device: str | torch.device = "cpu") -> list[tuple[str, float]]:
    """The k texts most similar to the query: embed it, dot with the normalised embeddings, take the top k."""
    q = embed_texts(encoder, tokenizer, [query], device, max_length=64)[0]      # (D,), unit length
    scores = embeddings @ q                                                   # cosine similarities, (N,)
    top = scores.topk(min(k, len(scores)))
    return [(texts[i], float(s)) for s, i in zip(top.values, top.indices)]


def clip_and_step(model: nn.Module, optimizer: torch.optim.Optimizer, max_grad_norm: float) -> float:
    """Clip the gradient norm, take the optimizer step, zero the gradients. Returns the norm before clipping.

    ``torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)`` rescales every gradient so their
    combined norm is at most max_grad_norm, and returns the norm it found. One oversized step can undo a
    pretrained model; clipping is the seat belt.
    """
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def fit_lm(model: nn.Module, train_loader, val_loader, epochs: int = 1, lr: float = 2e-5, accumulation_steps: int = 1,
           max_grad_norm: float = 1.0, device: str | torch.device = "cpu") -> dict:
    """Fine-tuning loop for Hugging Face classification models.

    AdamW. For each batch (a dict): move it to device, ``out = model(**batch)``, ``loss = out.loss / accumulation_steps``,
    backward; every ``accumulation_steps`` batches (and at the end of the epoch) call clip_and_step. Gradient
    accumulation makes several small batches act as one large one, which is how a 6 GB GPU trains a 66M-parameter
    model. After each epoch evaluate the mean validation loss and accuracy (``out.logits.argmax(-1)``).
    Returns {"train": [...], "val": [...], "val_acc": [...], "steps": total optimizer steps}.
    """
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"train": [], "val": [], "val_acc": [], "steps": 0}
    for _ in range(epochs):
        model.train()
        total, seen, pending = 0.0, 0, 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / accumulation_steps).backward()                    # gradients add up across the small batches
            pending += 1
            total += out.loss.item() * len(batch["labels"]); seen += len(batch["labels"])
            if pending == accumulation_steps or i == len(train_loader) - 1:
                clip_and_step(model, optimizer, max_grad_norm)
                history["steps"] += 1
                pending = 0
        history["train"].append(total / seen)
        model.eval()
        v_total, v_seen, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                v_total += out.loss.item() * len(batch["labels"]); v_seen += len(batch["labels"])
                correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
        history["val"].append(v_total / v_seen)
        history["val_acc"].append(correct / v_seen)
    return history
