# tests/test_normalization.py
"""Correctness tests for LayerNorm/RMSNorm, parametrized over backends."""

import pytest
import torch

from gluon_by_example._validation import check_normalization_inputs

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_validation_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        check_normalization_inputs(torch.randn(4, 8), torch.randn(8))


@requires_cuda
def test_validation_rejects_wrong_weight_len():
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(ValueError, match="length 8"):
        check_normalization_inputs(x, torch.randn(7, device="cuda"))


@requires_cuda
def test_validation_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        check_normalization_inputs(torch.randn(4, 8, 2, device="cuda"),
                                   torch.randn(8, device="cuda"))
