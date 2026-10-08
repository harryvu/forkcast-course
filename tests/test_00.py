"""Lesson 0 definition of done.

Run:  pytest tests/test_00.py -q
Expected on a CPU runtime: 7 passed, 1 skipped, 1 deselected.  On a GPU runtime: 8 passed, 1 deselected.
The deselected test is the stretch goal; the skipped one needs a GPU.
Stretch goal:  pytest tests/test_00.py -m stretch -q
"""

import pytest
import torch

from forkcast import env, tensors

fast = pytest.mark.fast  # the CI workflow runs `pytest -m fast`; stretch tests stay out of it


@fast
def test_env_report():
    info = env.report()
    assert info["torch_version"] == torch.__version__
    assert info["device"] in {"cpu", "cuda"}
    assert isinstance(info["cuda_available"], bool)
    assert info["python_version"].count(".") == 2


@fast
def test_to_batch():
    x = torch.randn(4)
    out = tensors.to_batch(x)
    assert out.shape == (1, 4), f"expected shape (1, 4), got {tuple(out.shape)}"
    assert torch.equal(out[0], x), "the numbers must be unchanged"


@fast
def test_flatten_images():
    images = torch.randn(8, 3, 28, 28)
    out = tensors.flatten_images(images)
    assert out.shape == (8, 3 * 28 * 28), f"expected shape (8, 2352), got {tuple(out.shape)}"
    assert torch.equal(out[0], images[0].reshape(-1)), "each row must be that image's pixels in memory order"


@fast
def test_channels_last():
    images = torch.randn(8, 3, 28, 28)
    out = tensors.channels_last(images)
    assert out.shape == (8, 28, 28, 3), f"expected shape (8, 28, 28, 3), got {tuple(out.shape)}"
    assert out[2, 5, 7, 1] == images[2, 1, 5, 7], "pixel (5, 7) of channel 1 must stay the same number"


@fast
def test_normalize_columns():
    x = torch.randn(32, 4) * torch.tensor([1.0, 10.0, 100.0, 0.5]) + torch.tensor([0.0, 5.0, -3.0, 2.0])
    out = tensors.normalize_columns(x)
    assert out.shape == (32, 4), f"expected shape (32, 4), got {tuple(out.shape)}"
    assert torch.allclose(out.mean(dim=0), torch.zeros(4), atol=1e-5), "every column should have mean 0"
    assert torch.allclose(out.std(dim=0, correction=0), torch.ones(4), atol=1e-4), "every column should have std 1"


@fast
def test_pairwise_distance():
    a = torch.randn(5, 3)
    b = torch.randn(7, 3)
    out = tensors.pairwise_distance(a, b)
    assert out.shape == (5, 7), f"expected shape (5, 7), got {tuple(out.shape)}"
    assert torch.allclose(out, torch.cdist(a, b), atol=1e-5), "distances must match torch.cdist"
    assert torch.allclose(tensors.pairwise_distance(a, a).diagonal(), torch.zeros(5), atol=1e-5), "a point is at distance 0 from itself"


@fast
def test_move_to_cpu():
    x = torch.randn(3)
    out = tensors.move_to(x, "cpu")
    assert out.device.type == "cpu"
    assert torch.equal(out, x)


@fast
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device; pick a GPU runtime in Colab to run it")
def test_move_to_cuda():
    x = torch.randn(3)
    out = tensors.move_to(x, "cuda")
    assert out.device.type == "cuda"
    assert torch.equal(out.cpu(), x)


@pytest.mark.stretch
def test_one_hot():
    idx = torch.tensor([0, 2, 1, 2])
    out = tensors.one_hot(idx, num_classes=3)
    assert out.shape == (4, 3), f"expected shape (4, 3), got {tuple(out.shape)}"
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.nn.functional.one_hot(idx, 3).float())
