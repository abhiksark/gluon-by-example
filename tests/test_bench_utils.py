# tests/test_bench_utils.py
"""Tests for shared benchmark helpers."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from bench_utils import device_arch, slugify  # noqa: E402


def test_slugify_device_name():
    assert slugify("NVIDIA RTX A6000") == "nvidia-rtx-a6000"


def test_slugify_strips_edge_punctuation():
    assert slugify("  GeForce RTX 5090!  ") == "geforce-rtx-5090"


def test_device_arch_matches_torch_backend():
    import torch

    arch = device_arch()
    if torch.version.hip is not None:
        assert arch == f"hip {torch.version.hip}"
    else:
        assert arch == f"cuda {torch.version.cuda}"
