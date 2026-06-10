# tests/test_validation.py
"""Tests for shared kernel input validation."""

import pytest
import torch

from gluon_by_example._validation import check_elementwise_inputs

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
def test_accepts_matching_cuda_tensors():
    x = torch.randn(16, device="cuda")
    y = torch.randn(16, device="cuda")
    check_elementwise_inputs(x, y)  # must not raise


def test_rejects_cpu_tensors():
    x = torch.randn(16)
    y = torch.randn(16)
    with pytest.raises(ValueError, match="CUDA"):
        check_elementwise_inputs(x, y)


@requires_cuda
def test_rejects_shape_mismatch():
    x = torch.randn(16, device="cuda")
    y = torch.randn(8, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        check_elementwise_inputs(x, y)


@requires_cuda
def test_rejects_dtype_mismatch():
    x = torch.randn(16, device="cuda", dtype=torch.float32)
    y = torch.randn(16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        check_elementwise_inputs(x, y)


@requires_cuda
def test_rejects_noncontiguous():
    x = torch.randn(16, 2, device="cuda")[:, 0]
    y = torch.randn(16, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        check_elementwise_inputs(x, y)
