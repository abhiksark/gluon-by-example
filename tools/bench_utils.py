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


def device_arch() -> str:
    """Returns the active GPU backend and its version, for bench provenance.

    PyTorch's ROCm build reuses the ``torch.cuda`` namespace, so the device
    name alone does not reveal whether a run used the CUDA or the HIP stack.
    This names it explicitly, e.g. ``cuda 13.0`` on NVIDIA or ``hip 6.2`` on
    AMD, provenance the device slug cannot carry.
    """
    if torch.version.hip is not None:
        return f"hip {torch.version.hip}"
    return f"cuda {torch.version.cuda}"
