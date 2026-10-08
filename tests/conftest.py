"""Shared pytest fixtures for the Forkcast course."""

import pytest
import torch


@pytest.fixture(autouse=True)
def fixed_seed():
    """Every test starts from the same random state, so failures are repeatable."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def device():
    """The best available device, the same way the course code picks it."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lesson_module(name: str, *attrs: str, lesson: str = ""):
    """Import ``forkcast.<name>`` for a lesson's tests, skipping the whole file until the
    lesson has been started. The notebook of each lesson creates the file with
    %%writefile, so a missing module or attribute simply means "not there yet"."""
    module = pytest.importorskip(f"forkcast.{name}", reason=f"{lesson or name}: not started yet (its notebook creates forkcast/{name}.py)")
    for attr in attrs:
        if not hasattr(module, attr):
            pytest.skip(f"{lesson or name}: forkcast/{name}.py has no {attr} yet; run the lesson's notebook to add it", allow_module_level=True)
    return module
