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
_dw_mode = "atomic"  # P4 adds "partial"


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


@triton.jit
def _layernorm_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
                             row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dy = tl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = tl.where(mask, w * dy, 0.0)
    c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
    c2 = tl.sum(wdy, axis=0) / n_cols
    dx = (wdy - x_hat * c1 - c2) * rstd
    tl.store(dx_ptr + row * row_stride + cols, dx, mask=mask)


@triton.jit
def _rmsnorm_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
                           row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dy = tl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    x_hat = tl.where(mask, x * rstd, 0.0)
    wdy = tl.where(mask, w * dy, 0.0)
    c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
    dx = (wdy - x_hat * c1) * rstd
    tl.store(dx_ptr + row * row_stride + cols, dx, mask=mask)


@triton.jit
def _ln_dw_atomic_kernel(dy_ptr, x_ptr, mean_ptr, rstd_ptr, dw_ptr, db_ptr,
                         row_stride, n_cols, BLOCK: tl.constexpr):
    # Floor: every program atomically adds its row's contribution to the
    # shared dw/db accumulators. Correct but contention-bound on N columns.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dy = tl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
    tl.atomic_add(dw_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(db_ptr + cols, dy, mask=mask)


@triton.jit
def _rms_dw_atomic_kernel(dy_ptr, x_ptr, rstd_ptr, dw_ptr,
                          row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dy = tl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    x_hat = tl.where(mask, x * rstd, 0.0)
    tl.atomic_add(dw_ptr + cols, dy * x_hat, mask=mask)


def _ln_dw_partial(dy, x, mean, rstd, dw, db):
    raise NotImplementedError  # implemented in P4


def _rms_dw_partial(dy, x, rstd, dw):
    raise NotImplementedError  # implemented in P4


def _ln_weight_grads(dy, x, mean, rstd, dtype):
    n_rows, n_cols = x.shape
    block, num_warps = _launch_meta(n_cols)
    dw = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
    db = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
    if _dw_mode == "atomic":
        _ln_dw_atomic_kernel[(n_rows,)](
            dy, x, mean, rstd, dw, db, x.stride(0), n_cols,
            BLOCK=block, num_warps=num_warps)
    else:
        _ln_dw_partial(dy, x, mean, rstd, dw, db)  # P4
    return dw.to(dtype), db.to(dtype)


def _rms_weight_grad(dy, x, rstd, dtype):
    n_rows, n_cols = x.shape
    block, num_warps = _launch_meta(n_cols)
    dw = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
    if _dw_mode == "atomic":
        _rms_dw_atomic_kernel[(n_rows,)](
            dy, x, rstd, dw, x.stride(0), n_cols, BLOCK=block, num_warps=num_warps)
    else:
        _rms_dw_partial(dy, x, rstd, dw)  # P4
    return dw.to(dtype)


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
        x, weight, mean, rstd = ctx.saved_tensors
        n_rows, n_cols = x.shape
        block, num_warps = _launch_meta(n_cols)
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        _layernorm_bwd_dx_kernel[(n_rows,)](
            dy, x, weight, mean, rstd, dx, x.stride(0), n_cols,
            BLOCK=block, num_warps=num_warps)
        dw, db = _ln_weight_grads(dy, x, mean, rstd, weight.dtype)
        return dx, dw, db, None


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
        x, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x.shape
        block, num_warps = _launch_meta(n_cols)
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        _rmsnorm_bwd_dx_kernel[(n_rows,)](
            dy, x, weight, rstd, dx, x.stride(0), n_cols,
            BLOCK=block, num_warps=num_warps)
        dw = _rms_weight_grad(dy, x, rstd, weight.dtype)
        return dx, dw, None


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim of a 2-D CUDA tensor (differentiable)."""
    return _LayerNorm.apply(x, weight, bias, eps)


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm over the last dim of a 2-D CUDA tensor (differentiable)."""
    return _RMSNorm.apply(x, weight, eps)
