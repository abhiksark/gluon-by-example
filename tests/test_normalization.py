# tests/test_normalization.py
"""Correctness tests for LayerNorm/RMSNorm, parametrized over backends."""

import pytest
import torch
import torch.nn.functional as F

from gluon_by_example._validation import check_normalization_inputs
from gluon_by_example.triton_impl import normalization as tn

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(params=["atomic", "partial"])
def dw_mode(request):
    prev = tn._dw_mode
    tn.set_dw_mode(request.param)
    yield request.param
    tn.set_dw_mode(prev)


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


TRITON = {"layer_norm": tn.layer_norm, "rms_norm": tn.rms_norm}
# (1,1) degenerate, (1823,781) padding mask, (4,1024) multi-warp, (16,64) tiny.
SHAPES = [(1, 1), (8, 256), (1823, 781), (4, 1024), (16, 64)]


def _rms_ref(x, w, eps):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), w, eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layernorm_forward_matches_torch(shape, dtype):
    m, n = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(n, device="cuda", dtype=dtype)
    b = torch.randn(n, device="cuda", dtype=dtype)
    out = tn.layer_norm(x, w, b)
    ref = F.layer_norm(x, (n,), w, b, eps=1e-5)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_forward_matches_torch(shape, dtype):
    m, n = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(n, device="cuda", dtype=dtype)
    out = tn.rms_norm(x, w)
    torch.testing.assert_close(out, _rms_ref(x, w, 1e-5), atol=1e-2, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize("shape", [(64, 256), (1823, 781), (16, 64)])
def test_layernorm_backward_matches_autograd(shape, dw_mode):
    m, n = shape
    x = torch.randn(shape, device="cuda", dtype=torch.float64, requires_grad=True)
    w = torch.randn(n, device="cuda", dtype=torch.float64, requires_grad=True)
    b = torch.randn(n, device="cuda", dtype=torch.float64, requires_grad=True)
    xr, wr, br = (t.detach().clone().requires_grad_(True) for t in (x, w, b))
    g = torch.randn(shape, device="cuda", dtype=torch.float64)
    tn.layer_norm(x, w, b).backward(g)
    F.layer_norm(xr, (n,), wr, br, eps=1e-5).backward(g)
    for a, e in ((x, xr), (w, wr), (b, br)):
        torch.testing.assert_close(a.grad, e.grad, atol=1e-6, rtol=1e-5)


@requires_cuda
@pytest.mark.parametrize("shape", [(64, 256), (1823, 781), (16, 64)])
def test_rmsnorm_backward_matches_autograd(shape, dw_mode):
    m, n = shape
    x = torch.randn(shape, device="cuda", dtype=torch.float64, requires_grad=True)
    w = torch.randn(n, device="cuda", dtype=torch.float64, requires_grad=True)
    xr, wr = (t.detach().clone().requires_grad_(True) for t in (x, w))
    g = torch.randn(shape, device="cuda", dtype=torch.float64)
    tn.rms_norm(x, w).backward(g)
    (_rms_ref(xr, wr, 1e-5)).backward(g)
    for a, e in ((x, xr), (w, wr)):
        torch.testing.assert_close(a.grad, e.grad, atol=1e-6, rtol=1e-5)
