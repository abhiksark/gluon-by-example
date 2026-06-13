# tests/test_matmul.py
"""Correctness tests for matmul, parametrized over backends."""

import pytest
import torch
import triton

from gluon_by_example.triton_impl.matmul import _CONFIGS, _matmul_kernel
from gluon_by_example.triton_impl.matmul import matmul as triton_matmul
from gluon_by_example.gluon_impl.matmul import matmul as gluon_matmul

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Chapter 5 adds the Gluon implementation here.
BACKENDS = {
    "triton": triton_matmul,
    "gluon": gluon_matmul,
}

# (M, K, N): degenerate single element, one tile, odd everything (masks on
# every edge), odd-and-larger, and a multi-tile rectangular case.
SHAPES = [(1, 1, 1), (16, 16, 16), (33, 77, 55), (255, 257, 129), (512, 1024, 768)]
# The gluon chapter kernel is tile-aligned fp16 only (M%128, N%128, K%64).
# (M, K, N): one tile, two tiles square, a rectangular multi-tile case.
ALIGNED_SHAPES = [(128, 64, 128), (256, 256, 256), (384, 128, 256)]

# Compare against the fp64 product of the same inputs — the mathematically
# true answer. cuBLAS output is itself ~1-2 output-ulps from that truth (on
# bf16, farther than this kernel), so it is too noisy to be the reference.
# Bounds: fp32 accumulation + one downcast stays within ~1 output ulp
# (fp16 ulp 2^-11, bf16 ulp 2^-8); atol covers cancellation near zero.
# Validated on the A6000: worst observed error/tolerance ratio is 0.24.
TOLS = {
    torch.float16: dict(rtol=2e-3, atol=1e-2),
    torch.bfloat16: dict(rtol=1.6e-2, atol=2e-2),
}


@requires_cuda
@pytest.mark.parametrize("backend", ["triton"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matches_ground_truth(backend, shape, dtype):
    m, k, n = shape
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    out = BACKENDS[backend](a, b)
    torch.testing.assert_close(out.double(), a.double() @ b.double(), **TOLS[dtype])


@requires_cuda
@pytest.mark.parametrize("config", _CONFIGS, ids=lambda c: str(c.kwargs))
def test_every_config_is_correct(config):
    # The autotuner asserts only its winner; this pins every listed config.
    m, k, n = 255, 257, 129
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(m, config.kwargs["BLOCK_M"])
            * triton.cdiv(n, config.kwargs["BLOCK_N"]),)
    _matmul_kernel.fn[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **config.kwargs, num_stages=config.num_stages, num_warps=config.num_warps,
    )
    torch.testing.assert_close(c.double(), a.double() @ b.double(), **TOLS[torch.float16])


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_invalid_inputs(backend):
    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randn(16, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA"):
        BACKENDS[backend](a, b)


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_inner_dim_mismatch(backend):
    a = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(8, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="inner"):
        BACKENDS[backend](a, b)


@requires_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_fp32(backend):
    a = torch.randn(8, 16, device="cuda")
    b = torch.randn(16, 4, device="cuda")
    with pytest.raises(ValueError, match="unsupported"):
        BACKENDS[backend](a, b)


@requires_cuda
@pytest.mark.parametrize("shape", ALIGNED_SHAPES)
def test_gluon_matches_ground_truth(shape):
    m, k, n = shape
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    out = gluon_matmul(a, b)
    torch.testing.assert_close(
        out.double(), a.double() @ b.double(), **TOLS[torch.float16])


@requires_cuda
def test_gluon_rejects_unaligned():
    a = torch.randn(130, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="tile-aligned"):
        gluon_matmul(a, b)


@requires_cuda
def test_gluon_rejects_bf16():
    a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float16"):
        gluon_matmul(a, b)
