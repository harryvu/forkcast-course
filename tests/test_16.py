"""Lesson 16 definition of done (the capstone).

Run:  pytest tests/test_16.py -q
Expected: 7 passed.
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module

cap = lesson_module("capstone", "FusionModel", "collate_listings", "run_ablation", "write_model_card", lesson="Lesson 16")
train = lesson_module("train", "fit_v2", lesson="Lesson 8")

fast = pytest.mark.fast


@pytest.fixture(scope="module")
def data():
    return cap.listing_loaders(n_train=1200, n_val=400, batch_size=64)


@fast
def test_listings_are_balanced_and_deterministic():
    images, texts, tabular, labels = cap.make_listings(400, seed=0)
    assert images.shape == (400, 3, 32, 32) and len(texts) == 400 and tabular.shape == (400, 3) and labels.shape == (400,)
    assert 0.45 < labels.float().mean().item() < 0.55, "the label is 'above the median demand': balanced by construction"
    assert isinstance(texts[0], str) and any(w in texts[0] for w in list(cap.PORTION) + [w.capitalize() for w in cap.PORTION])
    images2, texts2, tabular2, labels2 = cap.make_listings(400, seed=0)
    assert torch.equal(images, images2) and texts == texts2 and torch.equal(labels, labels2)


@fast
def test_collate_shapes_with_label_last(data):
    train_loader, _, vocab, _ = data
    images, ids, mask, tabular, labels = next(iter(train_loader))
    assert images.shape == (64, 3, 32, 32) and ids.shape == mask.shape and ids.dtype == torch.long and mask.dtype == torch.bool
    assert tabular.shape == (64, 3) and labels.shape == (64,) and labels.dtype == torch.long
    assert abs(tabular.mean().item()) < 0.3, "tabular features arrive normalized with the training statistics"


@fast
def test_fusion_model_shapes_and_ablation_switches(data):
    train_loader, _, vocab, _ = data
    images, ids, mask, tabular, labels = next(iter(train_loader))
    torch.manual_seed(0)
    full = cap.FusionModel(len(vocab)).eval()
    assert full(images, ids, mask, tabular).shape == (64, 2)
    assert full.encode_image(images).shape == (64, 32) and full.encode_text(ids, mask).shape == (64, 32)
    torch.manual_seed(0)
    tab_only = cap.FusionModel(len(vocab), modalities=("tabular",)).eval()
    out_a = tab_only(images, ids, mask, tabular)
    out_b = tab_only(torch.rand_like(images), ids, mask, tabular)
    assert torch.allclose(out_a, out_b), "a model built without the image modality must ignore the image"
    assert not torch.allclose(full(images, ids, mask, tabular), full(torch.rand_like(images), ids, mask, tabular)), "the full model does look at the image"


@fast
def test_fusion_beats_every_single_modality(data):
    train_loader, val_loader, vocab, _ = data
    make = lambda mods: cap.FusionModel(len(vocab), modalities=mods)
    results = cap.run_ablation(make, train_loader, val_loader, [("image",), ("text",), ("tabular",), ("image", "text", "tabular")], epochs=12, lr=2e-3)
    singles = {k: v for k, v in results.items() if len(k) == 1}
    fused = results[("image", "text", "tabular")]
    assert fused > 0.8, f"fused accuracy {fused:.2f}"
    assert fused > max(singles.values()) + 0.1, f"fusion {fused:.2f} should clearly beat the best single modality {singles}"
    assert all(0.0 <= v <= 1.0 for v in results.values())


@fast
def test_run_ablation_returns_every_combo(data):
    train_loader, val_loader, vocab, _ = data
    small_train = torch.utils.data.DataLoader(train_loader.dataset, batch_size=64, collate_fn=cap.collate_listings, sampler=range(128))
    results = cap.run_ablation(lambda m: cap.FusionModel(len(vocab), modalities=m), small_train, val_loader, [("text",), ("text", "tabular")], epochs=1)
    assert set(results) == {("text",), ("text", "tabular")}


@fast
def test_model_card_has_the_required_sections(tmp_path):
    path = tmp_path / "MODEL_CARD.md"
    text = cap.write_model_card(path, "Forkcast listing demand", "Rank new listings; not for pricing decisions.", "1,600 synthetic listings.",
                                {"accuracy (fused)": 0.9125, "accuracy (image only)": 0.6, "epochs": 8}, ["Synthetic data only.", "32×32 images."],
                                ["Lesson 7: image encoder", "Lesson 11: text encoder", "Lesson 4: tabular MLP"])
    assert path.read_text(encoding="utf-8") == text
    for section in ["## Intended use", "## Training data", "## Evaluation", "## Limitations", "## Provenance"]:
        assert section in text, f"missing {section}"
    assert text.index("## Intended use") < text.index("## Training data") < text.index("## Evaluation") < text.index("## Limitations") < text.index("## Provenance")
    assert "| accuracy (fused) | 0.912 |" in text and "| epochs | 8 |" in text
    assert "- Synthetic data only." in text and "- Lesson 7: image encoder" in text


@fast
def test_artifact_round_trip(data, tmp_path):
    train_loader, val_loader, vocab, (tab_mean, tab_std) = data
    torch.manual_seed(0)
    model = cap.FusionModel(len(vocab))
    train.fit_v2(model, train_loader, val_loader, epochs=1, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
    torch.save({"state_dict": model.state_dict(), "itos": vocab.itos, "tab_mean": tab_mean, "tab_std": tab_std, "d": 32, "hidden": 64, "size": 32, "modalities": list(model.modalities)}, tmp_path / "listing.pt")
    bundle = torch.load(tmp_path / "listing.pt", weights_only=True)
    fresh = cap.FusionModel(len(bundle["itos"]), d=bundle["d"], hidden=bundle["hidden"], size=bundle["size"], modalities=tuple(bundle["modalities"]))
    fresh.load_state_dict(bundle["state_dict"])
    images, ids, mask, tabular, _ = next(iter(val_loader))
    with torch.inference_mode():
        assert torch.equal(fresh.eval()(images, ids, mask, tabular), model.eval()(images, ids, mask, tabular))
