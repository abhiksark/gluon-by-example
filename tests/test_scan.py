# tests/test_scan.py
"""Correctness for the row-wise cumsum scan kernel."""

import pytest
import torch

from gluon_by_example.triton_impl.scan import cumsum

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 1), (8, 500), (1823, 781), (4, 4096)])
def test_cumsum_matches_torch(shape):
    x = torch.randn(shape, device="cuda")
    torch.testing.assert_close(cumsum(x), torch.cumsum(x, dim=-1), atol=1e-3, rtol=1e-3)
