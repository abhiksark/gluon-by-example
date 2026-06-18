# src/gluon_by_example/gluon_impl/normalization.py
"""LayerNorm and RMSNorm forward pass in Gluon (forward-only; backward in P6).

One program normalizes one full row held in registers, mirroring the Triton
twin in triton_impl/normalization.py. Statistics (mean, rstd) are saved by the
forward pass for the backward pass added in P6.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gluon_by_example._validation import check_normalization_inputs

_MAX_COLS = 32768


def _check_cols(x):
    """Return n_cols, raising on empty or oversized rows."""
    n_cols = x.shape[1]
    if n_cols == 0:
        raise ValueError("rows must be non-empty")
    if n_cols > _MAX_COLS:
        raise ValueError(f"n_cols={n_cols} exceeds fused-row limit {_MAX_COLS}")
    return n_cols


def _launch_meta(n_cols):
    """Block size and warp count for a fused row, matching the Triton twin."""
    block = triton.next_power_of_2(n_cols)
    num_warps = 4
    if block >= 2048:
        num_warps = 8
    if block >= 8192:
        num_warps = 16
    return block, num_warps


# Reduction helpers in Gluon style: spelled-out combine_fn pairs so the reader
# sees exactly what each reduction computes.
@gluon.jit
def _add_fn(a, b):
    return a + b


# LayerNorm forward kernel.
#
# Each program owns one row. The row is loaded once into float32, mean and
# variance are reduced via gl.reduce, then the normalized + affine output is
# written once. mean and rstd are stored to per-row scratch buffers for the
# backward pass (P6).
#
# gl.store of a scalar (mean_ptr + row, rstd_ptr + row): these are scalar
# pointer offsets. Gluon's gl.store accepts a scalar tensor produced by
# gl.reduce; whether it needs explicit reshape is unverified without a GPU run
# -- flagged as UNVERIFIED_SCALAR_STORE below.
@gluon.jit
def _layernorm_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rstd_ptr,
                          row_stride, n_cols, eps,
                          BLOCK: gl.constexpr, layout: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    x = gl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    x = x.to(gl.float32)
    # Mean: sum over the row then divide by the true column count.
    mean = gl.reduce(x, axis=0, combine_fn=_add_fn) / n_cols
    xc = x - mean  # centered; padded lanes are zero so they don't corrupt var
    # Mask xc so padded lanes do not contribute to variance.
    xc = gl.where(mask, xc, 0.0)
    var = gl.reduce(xc * xc, axis=0, combine_fn=_add_fn) / n_cols
    rstd = 1.0 / gl.sqrt(var + eps)
    w = gl.load(w_ptr + cols, mask=mask, other=0.0).to(gl.float32)
    b = gl.load(b_ptr + cols, mask=mask, other=0.0).to(gl.float32)
    y = xc * rstd * w + b
    # UNVERIFIED_SCALAR_STORE: gl.store of a scalar (mean, rstd) at a scalar
    # pointer offset. Modeled on tl.store(mean_ptr + row, mean) from the
    # Triton twin; Gluon's scalar-store behavior needs a GPU run to confirm.
    gl.store(mean_ptr + row, mean)
    gl.store(rstd_ptr + row, rstd)
    gl.store(y_ptr + row * row_stride + cols, y, mask=mask)


# RMSNorm forward kernel.
#
# Same structure as LayerNorm but without mean subtraction: the mean-square is
# computed directly from x, not from centered x.
@gluon.jit
def _rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr,
                        row_stride, n_cols, eps,
                        BLOCK: gl.constexpr, layout: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    x = gl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    x = x.to(gl.float32)
    # Mask x so padded lanes do not inflate the mean-square.
    xm = gl.where(mask, x, 0.0)
    ms = gl.reduce(xm * xm, axis=0, combine_fn=_add_fn) / n_cols
    rstd = 1.0 / gl.sqrt(ms + eps)
    w = gl.load(w_ptr + cols, mask=mask, other=0.0).to(gl.float32)
    y = x * rstd * w
    # UNVERIFIED_SCALAR_STORE: same caveat as LayerNorm above.
    gl.store(rstd_ptr + row, rstd)
    gl.store(y_ptr + row * row_stride + cols, y, mask=mask)


class _LayerNorm(torch.autograd.Function):
    """torch.autograd.Function wrapping the Gluon LayerNorm forward kernel."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        check_normalization_inputs(x, weight, bias)
        n_cols = _check_cols(x)
        n_rows = x.shape[0]
        block, num_warps = _launch_meta(n_cols)
        # BlockedLayout: same dtype-width coalescing formula as gluon_impl/softmax.py.
        size_per_thread = max(
            min(block // (32 * num_warps), 16 // x.element_size()), 1
        )
        layout = gl.BlockedLayout(
            size_per_thread=[size_per_thread],
            threads_per_warp=[32],
            warps_per_cta=[num_warps],
            order=[0],
        )
        y = torch.empty_like(x)
        mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        _layernorm_fwd_kernel[(n_rows,)](
            x, weight, bias, y, mean, rstd,
            x.stride(0), n_cols, eps,
            BLOCK=block, layout=layout, num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("LayerNorm backward not yet implemented (P6)")


class _RMSNorm(torch.autograd.Function):
    """torch.autograd.Function wrapping the Gluon RMSNorm forward kernel."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        check_normalization_inputs(x, weight)
        n_cols = _check_cols(x)
        n_rows = x.shape[0]
        block, num_warps = _launch_meta(n_cols)
        size_per_thread = max(
            min(block // (32 * num_warps), 16 // x.element_size()), 1
        )
        layout = gl.BlockedLayout(
            size_per_thread=[size_per_thread],
            threads_per_warp=[32],
            warps_per_cta=[num_warps],
            order=[0],
        )
        y = torch.empty_like(x)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        _rmsnorm_fwd_kernel[(n_rows,)](
            x, weight, y, rstd,
            x.stride(0), n_cols, eps,
            BLOCK=block, layout=layout, num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("RMSNorm backward not yet implemented (P6)")


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim of a 2-D CUDA tensor (forward only).

    Args:
        x: 2-D, contiguous CUDA tensor (float16/bfloat16/float32/float64).
        weight: 1-D CUDA tensor of length x.shape[1], same dtype as x.
        bias: 1-D CUDA tensor of length x.shape[1], same dtype as x.
        eps: Numerical stability epsilon. Defaults to 1e-5.

    Returns:
        Normalized tensor, same shape and dtype as x.
    """
    return _LayerNorm.apply(x, weight, bias, eps)


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm over the last dim of a 2-D CUDA tensor (forward only).

    Args:
        x: 2-D, contiguous CUDA tensor (float16/bfloat16/float32/float64).
        weight: 1-D CUDA tensor of length x.shape[1], same dtype as x.
        eps: Numerical stability epsilon. Defaults to 1e-5.

    Returns:
        Normalized tensor, same shape and dtype as x.
    """
    return _RMSNorm.apply(x, weight, eps)
