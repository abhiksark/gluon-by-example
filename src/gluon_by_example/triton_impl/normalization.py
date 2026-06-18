# src/gluon_by_example/triton_impl/normalization.py
"""LayerNorm and RMSNorm (forward + backward) in standard Triton.

One program normalizes one full row held in registers, the same fused-row
strategy as softmax. Statistics (mean, rstd) are saved on the forward pass and
reused on the backward pass. The weight gradient is a cross-row reduction,
offered two ways: an atomic_add floor and a two-stage grouped-partial climb.
"""

import torch
import triton
import triton.language as tl

from gluon_by_example._validation import check_normalization_inputs

_MAX_COLS = 32768


def _launch_meta(n_cols):
    """Block size and warp count for a fused row, matching softmax."""
    block = triton.next_power_of_2(n_cols)
    num_warps = 4
    if block >= 2048:
        num_warps = 8
    if block >= 8192:
        num_warps = 16
    return block, num_warps


@triton.jit
def _layernorm_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr,
                          row_stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    tl.store(y_ptr + row * row_stride + cols, y, mask=mask)


@triton.jit
def _rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr,
                        row_stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(rstd_ptr + row, rstd)
    tl.store(y_ptr + row * row_stride + cols, y, mask=mask)


def _check_cols(x):
    n_cols = x.shape[1]
    if n_cols == 0:
        raise ValueError("rows must be non-empty")
    if n_cols > _MAX_COLS:
        raise ValueError(f"n_cols={n_cols} exceeds fused-row limit {_MAX_COLS}")
    return n_cols


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        check_normalization_inputs(x, weight, bias)
        n_cols = _check_cols(x)
        n_rows = x.shape[0]
        block, num_warps = _launch_meta(n_cols)
        y = torch.empty_like(x)
        mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        _layernorm_fwd_kernel[(n_rows,)](
            x, weight, bias, y, mean, rstd, x.stride(0), n_cols, eps,
            BLOCK=block, num_warps=num_warps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError  # filled in P3


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        check_normalization_inputs(x, weight)
        n_cols = _check_cols(x)
        n_rows = x.shape[0]
        block, num_warps = _launch_meta(n_cols)
        y = torch.empty_like(x)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        _rmsnorm_fwd_kernel[(n_rows,)](
            x, weight, y, rstd, x.stride(0), n_cols, eps,
            BLOCK=block, num_warps=num_warps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError  # filled in P3


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim of a 2-D CUDA tensor (differentiable)."""
    return _LayerNorm.apply(x, weight, bias, eps)


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm over the last dim of a 2-D CUDA tensor (differentiable)."""
    return _RMSNorm.apply(x, weight, eps)
