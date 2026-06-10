# tools/bench_utils.py
"""Shared helpers for chapter benchmark scripts."""

import re

import torch


def slugify(name: str) -> str:
    """Returns a filesystem-safe slug for a device name.

    Args:
        name: Raw device name, e.g. ``NVIDIA RTX A6000``.

    Returns:
        Lowercase alphanumeric slug with hyphens replacing non-alphanumeric
        runs, e.g. ``nvidia-rtx-a6000``.
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def gpu_slug() -> str:
    """Returns the slug of CUDA device 0, e.g. ``nvidia-rtx-a6000``."""
    return slugify(torch.cuda.get_device_name(0))
