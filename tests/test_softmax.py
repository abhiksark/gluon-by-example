# tests/test_softmax.py
"""Correctness tests for softmax, parametrized over backends."""

import pytest
import torch

from gluon_by_example.gluon_impl.softmax import softmax as gluon_softmax
from gluon_by_example.triton_impl.softmax import softmax as triton_softmax

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BACKENDS = {
    "triton": triton_softmax,
    "gluon": gluon_softmax,
}

# Irregular shapes on purpose: (1823, 781) exercises the padding mask,
# (1, 1) the degenerate row, (4096, 4096) a large power-of-2 row,
# (4, 32768) covers the 16-warp path and the column-cap boundary.
# (16, 64) makes the Gluon layout larger than the row (lane replication).
SHAPES = [(1, 1), (8, 1024), (1823, 781), (4096, 4096), (4, 32768), (16, 64)]


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_matches_torch(backend, shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    out = BACKENDS[backend](x)
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_numerically_stable_at_large_magnitudes(backend):
    # Without the max-subtraction trick, exp() overflows on inputs this size.
    x = torch.randn(64, 781, device="cuda") * 1000
    out = BACKENDS[backend](x)
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_sum_to_one(backend):
    x = torch.randn(64, 500, device="cuda")
    out = BACKENDS[backend](x)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(64, device="cuda"))


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_invalid_inputs(backend):
    with pytest.raises(ValueError, match="CUDA"):
        BACKENDS[backend](torch.randn(4, 4))


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_too_many_cols(backend):
    x = torch.randn(1, 65536, device="cuda")
    with pytest.raises(ValueError, match="exceeds"):
        BACKENDS[backend](x)


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_empty_rows(backend):
    x = torch.empty(5, 0, device="cuda")
    with pytest.raises(ValueError, match="non-empty"):
        BACKENDS[backend](x)
