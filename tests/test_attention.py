# tests/test_attention.py
"""Correctness tests for FlashAttention, parametrized over backends."""

import pytest
import torch
import torch.nn.functional as F

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


from gluon_by_example.triton_impl import attention as ta  # noqa: E402

SHAPES = [(2, 2, 128, 64), (1, 1, 200, 64), (2, 4, 64, 32)]


def _ref(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
def test_triton_forward_matches_sdpa(shape, causal):
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))
    out = ta.attention(q, k, v, causal=causal)
    ref = _ref(q, k, v, causal)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
