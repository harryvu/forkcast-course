"""Lesson 8 definition of done.

Run:  pytest tests/test_08.py -q
Expected: 7 passed.
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module

train = lesson_module("train", "fit_v2", "train_one_epoch_amp", "save_checkpoint", "load_checkpoint", "make_scaler", lesson="Lesson 8")
vision = lesson_module("vision", "AugmentedTensorDataset", "train_transform", lesson="Lesson 8")
trips = lesson_module("trips", "make_loaders", lesson="Lesson 6")
TabularNet = lesson_module("model", "TabularNet", lesson="Lesson 6").TabularNet
Normalizer = lesson_module("preprocess", "Normalizer", lesson="Lesson 3").Normalizer

fast = pytest.mark.fast


@pytest.fixture(scope="module")
def loaders():
    frame = trips.load_trips(verbose=False)
    train_frame, val_frame, _ = trips.split_frame(frame, seed=0)
    norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
    return trips.make_loaders(train_frame, val_frame, norm, batch_size=512)


def fresh_model():
    torch.manual_seed(0)
    return TabularNet(4, hidden=(32,))


@fast
def test_make_scaler_only_on_cuda():
    assert train.make_scaler("cpu", amp=True) is None, "bfloat16 on CPU needs no loss scaling"
    assert train.make_scaler("cpu", amp=False) is None
    if torch.cuda.is_available():
        assert train.make_scaler("cuda", amp=True) is not None


@fast
def test_amp_epoch_runs_on_cpu_with_bfloat16(loaders):
    train_loader, val_loader = loaders
    model = fresh_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    before = train.evaluate_loader(model, val_loader, nn.MSELoss())
    loss = train.train_one_epoch_amp(model, train_loader, opt, nn.MSELoss(), device="cpu", amp=True, scaler=None)
    assert isinstance(loss, float) and torch.isfinite(torch.tensor(loss)), "autocast in bfloat16 must give a finite float32 loss"
    assert train.evaluate_loader(model, val_loader, nn.MSELoss()) < before, "one mixed-precision epoch should still learn"


@fast
def test_checkpoint_round_trip(loaders, tmp_path):
    train_loader, val_loader = loaders
    model = fresh_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    train.train_one_epoch_loader(model, train_loader, opt, nn.MSELoss())
    history = {"train": [1.0], "val": [2.0], "best_epoch": 0, "best_val": 2.0, "epoch_seconds": [0.1]}
    path = tmp_path / "ckpt.pt"
    train.save_checkpoint(path, model, opt, epoch=0, history=history, best_val=2.0)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"model", "optimizer", "epoch", "history", "best_val"}, f"checkpoint keys: {sorted(raw)}"
    other = fresh_model(); other_opt = torch.optim.Adam(other.parameters(), lr=3e-3)
    ckpt = train.load_checkpoint(path, other, other_opt)
    assert ckpt["epoch"] == 0 and ckpt["history"]["val"] == [2.0]
    x = next(iter(val_loader))
    assert torch.equal(other(x[0], x[1]), model(x[0], x[1])), "the reloaded model must predict identically"
    assert other_opt.state_dict()["state"].keys() == opt.state_dict()["state"].keys(), "the optimizer's running averages must be restored too"


@fast
def test_resume_continues_exactly_like_an_uninterrupted_run(loaders, tmp_path):
    train_loader, val_loader = loaders
    full = train.fit_v2(fresh_model(), train_loader, val_loader, epochs=5, lr=3e-3, ckpt_path=tmp_path / "full.pt")
    ckpt = tmp_path / "resume.pt"
    first = train.fit_v2(fresh_model(), train_loader, val_loader, epochs=3, lr=3e-3, ckpt_path=ckpt)
    assert len(first["train"]) == 3
    resumed = train.fit_v2(fresh_model(), train_loader, val_loader, epochs=5, lr=3e-3, ckpt_path=ckpt, resume=True)
    assert len(resumed["train"]) == 5, "resuming must continue the loaded history, not start a new one"
    assert torch.allclose(torch.tensor(resumed["train"]), torch.tensor(full["train"]), rtol=1e-4), f"resumed {resumed['train']} vs uninterrupted {full['train']}"
    assert torch.allclose(torch.tensor(resumed["val"]), torch.tensor(full["val"]), rtol=1e-4)


@fast
def test_fit_v2_records_timing_and_keeps_best_weights(loaders, tmp_path):
    train_loader, val_loader = loaders
    model = fresh_model()
    h = train.fit_v2(model, train_loader, val_loader, epochs=4, lr=3e-3, amp=True, device="cpu")
    assert len(h["epoch_seconds"]) == 4 and all(s > 0 for s in h["epoch_seconds"])
    assert abs(train.evaluate_loader(model, val_loader, nn.MSELoss()) - h["best_val"]) < 1e-4


@fast
def test_augmentation_changes_pixels_not_shapes():
    images, labels = vision.make_shapes(16, seed=0)
    plain = vision.AugmentedTensorDataset(images, labels, transform=None)
    x, y = plain[3]
    assert torch.equal(x, images[3]) and y == labels[3], "with no transform items must be the stored tensors"
    aug = vision.AugmentedTensorDataset(images, labels, transform=vision.train_transform(32))
    torch.manual_seed(1); a, ya = aug[3]
    torch.manual_seed(2); b, yb = aug[3]
    assert a.shape == (3, 32, 32) and b.shape == (3, 32, 32) and ya == yb == labels[3]
    assert not torch.equal(a, b), "two fetches of the same image should be two different random views"
    assert len(aug) == 16


@fast
def test_tensorboard_logging_writes_an_events_file(loaders, tmp_path):
    pytest.importorskip("tensorboard", reason="pip install tensorboard (present in Colab)")
    train_loader, val_loader = loaders
    train.fit_v2(fresh_model(), train_loader, val_loader, epochs=2, lr=3e-3, log_dir=tmp_path / "runs")
    files = list((tmp_path / "runs").glob("events.out.tfevents.*"))
    assert files, "SummaryWriter should have written an events file into log_dir"
