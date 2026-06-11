# tests/test_validation.py
"""Tests for shared kernel input validation."""

import pytest
import torch

from gluon_by_example._validation import (
    _MATMUL_DTYPES,
    _SOFTMAX_DTYPES,
    check_elementwise_inputs,
    check_matmul_inputs,
    check_softmax_inputs,
)

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


@requires_cuda
@pytest.mark.parametrize("dtype", _MATMUL_DTYPES)
def test_matmul_accepts_tensor_core_dtypes(dtype):
    a = torch.randn(8, 16, device="cuda", dtype=dtype)
    b = torch.randn(16, 4, device="cuda", dtype=dtype)
    check_matmul_inputs(a, b)  # must not raise


def test_matmul_rejects_cpu_tensors():
    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randn(16, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        check_matmul_inputs(a, b)


@requires_cuda
def test_matmul_rejects_non_2d():
    a = torch.randn(2, 8, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(16, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="2-D"):
        check_matmul_inputs(a, b)


@requires_cuda
def test_matmul_rejects_dtype_mismatch():
    a = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype mismatch"):
        check_matmul_inputs(a, b)


@requires_cuda
def test_matmul_rejects_fp32():
    a = torch.randn(8, 16, device="cuda")
    b = torch.randn(16, 4, device="cuda")
    with pytest.raises(ValueError, match="unsupported"):
        check_matmul_inputs(a, b)


@requires_cuda
def test_matmul_rejects_inner_dim_mismatch():
    a = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(8, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="inner"):
        check_matmul_inputs(a, b)


@requires_cuda
def test_matmul_rejects_empty_dims():
    a = torch.empty(0, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(16, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="non-empty"):
        check_matmul_inputs(a, b)


@requires_cuda
def test_matmul_rejects_noncontiguous():
    a = torch.randn(16, 8, device="cuda", dtype=torch.float16).t()
    b = torch.randn(16, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="contiguous"):
        check_matmul_inputs(a, b)
