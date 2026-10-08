"""Lesson 0. Environment check: which Python, which PyTorch, which device.

Run it as a module:  python -m forkcast.env
"""

from __future__ import annotations

import platform

import torch


def pick_device() -> torch.device:
    """Return the best available device: CUDA if present, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def report() -> dict:
    """Collect the facts the environment check prints. Also used by the tests."""
    device = pick_device()
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    return info


def main() -> None:
    info = report()
    print("Forkcast environment check")
    print(f"  python   {info['python_version']}")
    print(f"  torch    {info['torch_version']}")
    if info["cuda_available"]:
        print(f"  device   cuda     {info['device_name']}")
    else:
        print("  device   cpu      no CUDA device found; fine until Lesson 7")


if __name__ == "__main__":
    main()
