"""Lesson 6 definition of done.

Run:  pytest tests/test_06.py -q
Expected: 7 passed.
"""

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from conftest import lesson_module

trips = lesson_module("trips", "TripsDataset", "make_loaders", lesson="Lesson 6")
train = lesson_module("train", "fit_loader", lesson="Lesson 6")
TabularNet = lesson_module("model", "TabularNet", lesson="Lesson 6").TabularNet
MLP = lesson_module("model", "MLP", lesson="Lesson 4").MLP
Normalizer = lesson_module("preprocess", "Normalizer", lesson="Lesson 3").Normalizer

fast = pytest.mark.fast


@pytest.fixture(scope="module")
def frames():
    frame = trips.load_trips(verbose=False)
    return trips.split_frame(frame, seed=0)


@fast
def test_clean_trips_removes_impossible_rows():
    raw = pd.read_csv(trips.SAMPLE)
    frame = trips.clean_trips(raw, verbose=False)
    assert 0 < len(frame) < len(raw), "the sample contains impossible rows that must be removed"
    assert frame["minutes"].between(1, 120).all() and frame["trip_distance"].between(0, 50).all()
    assert frame["pu_zone"].between(0, 265).all() and frame["do_zone"].between(0, 265).all()
    assert frame["passenger_count"].notna().all()
    assert list(frame.columns) == trips.CONTINUOUS + trips.CATEGORICAL + [trips.TARGET]


@fast
def test_dataset_length_and_item_types(frames):
    train_frame, _, _ = frames
    norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
    ds = trips.TripsDataset(train_frame, norm)
    assert len(ds) == len(train_frame)
    cont, cat, target = ds[0]
    assert cont.shape == (4,) and cont.dtype == torch.float32
    assert cat.shape == (2,) and cat.dtype == torch.long, "zone ids are indices and must be int64 (long)"
    assert target.shape == () and target.dtype == torch.float32
    assert torch.allclose(ds.cont.mean(dim=0), torch.zeros(4), atol=0.05), "continuous features should be normalized with the fitted Normalizer"


@fast
def test_loader_batches_and_drop_last(frames):
    train_frame, val_frame, _ = frames
    norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
    train_loader, val_loader = trips.make_loaders(train_frame, val_frame, norm, batch_size=256)
    cont, cat, target = next(iter(train_loader))
    assert cont.shape == (256, 4) and cat.shape == (256, 2) and target.shape == (256,)
    assert len(train_loader) == len(train_frame) // 256, "drop_last=True: only full batches for training"
    assert train_loader.drop_last and not val_loader.drop_last
    assert sum(len(b[2]) for b in val_loader) == len(val_frame), "validation must see every row, including the short last batch"


@fast
def test_shuffle_changes_order_not_content(frames):
    train_frame, val_frame, _ = frames
    norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
    train_loader, val_loader = trips.make_loaders(train_frame, val_frame, norm, batch_size=64)
    torch.manual_seed(0); first = next(iter(train_loader))[2]
    torch.manual_seed(1); second = next(iter(train_loader))[2]
    assert not torch.equal(first, second), "shuffle=True should give a different first batch each epoch"
    val_targets = torch.cat([b[2] for b in val_loader])
    assert torch.equal(val_targets, torch.tensor(val_frame[trips.TARGET].values, dtype=torch.float32)), "an unshuffled loader keeps the original order"


@fast
def test_tabular_net_shapes():
    model = TabularNet(n_continuous=4, zone_vocab=266, emb_dim=8, hidden=(32,))
    assert model.pu.weight.shape == (266, 8) and model.do.weight.shape == (266, 8)
    x_cont = torch.randn(10, 4)
    x_cat = torch.randint(0, 266, (10, 2))
    assert model.pu(x_cat[:, 0]).shape == (10, 8), "embedding an (N,) index tensor gives (N, emb_dim)"
    assert model(x_cont, x_cat).shape == (10,)
    assert model.mlp.net[0].in_features == 4 + 2 * 8


@fast
def test_fit_loader_trains_on_the_sample(frames):
    train_frame, val_frame, _ = frames
    norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
    train_loader, val_loader = trips.make_loaders(train_frame, val_frame, norm, batch_size=256)
    torch.manual_seed(0)
    history = train.fit_loader(TabularNet(4), train_loader, val_loader, epochs=5, lr=3e-3)
    assert set(history) == {"train", "val", "best_epoch", "best_val"} and len(history["val"]) == 5
    assert history["val"][-1] < 0.5 * history["val"][0], f"five epochs should cut the validation loss a lot: {history['val'][0]:.1f} -> {history['val'][-1]:.1f}"


@fast
def test_zone_embeddings_beat_continuous_features_alone(frames):
    train_frame, val_frame, _ = frames
    norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
    train_loader, val_loader = trips.make_loaders(train_frame, val_frame, norm, batch_size=256)

    class ContinuousOnly(nn.Module):              # same MLP, zones ignored
        def __init__(self):
            super().__init__()
            self.mlp = MLP(4, (64, 64))
        def forward(self, x_cont, x_cat):
            return self.mlp(x_cont)

    torch.manual_seed(0)
    without = train.fit_loader(ContinuousOnly(), train_loader, val_loader, epochs=25, lr=3e-3)["best_val"]
    torch.manual_seed(0)
    with_zones = train.fit_loader(TabularNet(4), train_loader, val_loader, epochs=25, lr=3e-3)["best_val"]
    assert with_zones < 0.9 * without, f"zones carry information: with embeddings {with_zones:.2f}, without {without:.2f}"
