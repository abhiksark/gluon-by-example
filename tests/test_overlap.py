# tests/test_overlap.py
"""Correctness of the overlap demos: same result as the serial baseline."""

import pytest
import torch

from overlap_demo import graphed_repeat, overlapped_add, serial_add

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
def test_overlapped_matches_serial_and_truth():
    n = 1 << 20
    a = torch.randn(n, pin_memory=True)
    b = torch.randn(n, pin_memory=True)
    ser = serial_add(a, b)
    ovl = overlapped_add(a, b, n_chunks=8)
    torch.testing.assert_close(ovl, ser)
    torch.testing.assert_close(ovl, a + b)


@requires_cuda
def test_overlapped_handles_ragged_chunks():
    # n not divisible by n_chunks exercises the last short chunk.
    n = (1 << 20) + 7
    a = torch.randn(n, pin_memory=True)
    b = torch.randn(n, pin_memory=True)
    torch.testing.assert_close(overlapped_add(a, b, n_chunks=8), a + b)


@requires_cuda
def test_graphed_matches_eager():
    n = 1 << 16
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    torch.testing.assert_close(graphed_repeat(a, b, iters=10), a + b)
