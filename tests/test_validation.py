# tests/test_validation.py
"""Tests for shared kernel input validation."""

import pytest
import torch

from gluon_by_example._validation import check_elementwise_inputs, check_softmax_inputs, _SOFTMAX_DTYPES

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


@requires_cuda
def test_softmax_accepts_2d_cuda_tensor():
    x = torch.randn(8, 128, device="cuda")
    check_softmax_inputs(x)  # must not raise


def test_softmax_rejects_cpu_tensor():
    with pytest.raises(ValueError, match="CUDA"):
        check_softmax_inputs(torch.randn(8, 128))


@requires_cuda
def test_softmax_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        check_softmax_inputs(torch.randn(8, 4, 4, device="cuda"))


@requires_cuda
def test_softmax_rejects_integer_dtype():
    x = torch.ones(8, 128, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="floating"):
        check_softmax_inputs(x)


@requires_cuda
def test_softmax_rejects_noncontiguous():
    x = torch.randn(128, 8, device="cuda").t()
    with pytest.raises(ValueError, match="contiguous"):
        check_softmax_inputs(x)


@requires_cuda
@pytest.mark.parametrize("dtype", _SOFTMAX_DTYPES)
def test_softmax_accepts_supported_float_dtypes(dtype):
    x = torch.randn(8, 128, device="cuda").to(dtype)
    check_softmax_inputs(x)  # must not raise


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_softmax_rejects_fp8(dtype):
    x = torch.randn(8, 128, device="cuda").to(dtype)
    with pytest.raises(ValueError, match="unsupported"):
        check_softmax_inputs(x)
