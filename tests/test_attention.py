# tests/test_attention.py
"""Correctness tests for FlashAttention, parametrized over backends."""

import pytest
import torch

from gluon_by_example._validation import check_attention_inputs

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_validation_rejects_cpu():
    q = torch.randn(1, 1, 8, 64)
    with pytest.raises(ValueError, match="CUDA"):
        check_attention_inputs(q, q, q)


@requires_cuda
def test_validation_rejects_bad_headdim():
    q = torch.randn(1, 1, 8, 48, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        check_attention_inputs(q, q, q)


@requires_cuda
def test_validation_rejects_shape_mismatch():
    q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 9, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="shape"):
        check_attention_inputs(q, k, q)
