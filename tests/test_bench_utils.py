# tests/test_bench_utils.py
"""Tests for shared benchmark helpers."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from bench_utils import slugify  # noqa: E402


def test_slugify_device_name():
    assert slugify("NVIDIA RTX A6000") == "nvidia-rtx-a6000"


def test_slugify_strips_edge_punctuation():
    assert slugify("  GeForce RTX 5090!  ") == "geforce-rtx-5090"
