# tests/test_vector_add.py
"""Correctness tests for vector_add, parametrized over backends."""

import pytest
import torch

from gluon_by_example.triton_impl.vector_add import vector_add as triton_vector_add
from gluon_by_example.gluon_impl.vector_add import vector_add as gluon_vector_add

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BACKENDS = {
    "triton": triton_vector_add,
    "gluon": gluon_vector_add,
}


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1, 1024, 9999, 1 << 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matches_torch(backend, n, dtype):
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    out = BACKENDS[backend](x, y)
    torch.testing.assert_close(out, x + y)


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_invalid_inputs(backend):
    x = torch.randn(8)
    y = torch.randn(8)
    with pytest.raises(ValueError, match="CUDA"):
        BACKENDS[backend](x, y)
