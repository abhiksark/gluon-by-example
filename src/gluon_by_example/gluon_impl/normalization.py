# src/gluon_by_example/gluon_impl/normalization.py
"""LayerNorm and RMSNorm (forward + backward) in Gluon.

One program normalizes one full row held in registers, mirroring the Triton
twin in triton_impl/normalization.py. Statistics (mean, rstd) are saved by the
forward pass and reused on the backward pass.

Backward notes
--------------
dx: computed with gl.reduce for the two row-level scalars (c1, c2 for LN;
    c1 only for RMS), then elementwise arithmetic -- same math as the Triton
    twin, just spelled with gl.reduce + combine_fn.

dw / db (weight gradients): two-stage grouped-partial only -- no atomic_add.
    Stage 1 (_*_dw_partial_kernel): GROUP_M programs each accumulate a
    strided subset of rows (g, g+GROUP_M, g+2*GROUP_M, ...) into a
    [GROUP_M, n_cols] partial buffer using a while loop.
    Stage 2 (_dw_reduce_kernel): sum the partial buffer's rows into [n_cols].
    This is the climb form; the atomic floor stays Triton-only.

UNVERIFIED calls (flagged inline; confirm on a GPU run):
  - UNVERIFIED_SCALAR_STORE: gl.store of a scalar (mean, rstd) at a scalar
    pointer offset -- forward-only pattern, not changed here.
  - UNVERIFIED_SCALAR_LOAD: gl.load of a scalar (mean, rstd) inside the
    backward kernels (same pattern as forward, mirroring Triton's
    tl.load(mean_ptr + row)).
  - UNVERIFIED_WHILE_LOOP: dynamic while loop inside @gluon.jit -- mirrors
    the Triton twin's while-loop in partial kernels; Gluon's lowering of
    dynamic while loops (not for-range) is unverified without a GPU run.
  - UNVERIFIED_2D_PARTIAL_STORE: gl.store into a [group, n_cols] flat buffer
    at g * n_cols + cols -- 2-D partial buffer indexed linearly; unverified
    without a GPU run.
  - UNVERIFIED_REDUCE_FOR_LOOP: Python-range for loop inside @gluon.jit
    (_dw_reduce_kernel) -- mirrors Triton's for g in range(group_m); Gluon
    may or may not support a dynamic-upper-bound for-range at JIT time.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gluon_by_example._validation import check_normalization_inputs

_MAX_COLS = 32768
_GROUP_M = 64  # partial accumulator rows; mirrors Triton twin


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


# Reduction helpers in Gluon style: spelled-out combine_fn so the reader
# sees exactly what each reduction computes.
@gluon.jit
def _add_fn(a, b):
    return a + b


# ---------------------------------------------------------------------------
# Forward kernels
# ---------------------------------------------------------------------------

# LayerNorm forward kernel.
#
# Each program owns one row. The row is loaded once into float32, mean and
# variance are reduced via gl.reduce, then the normalized + affine output is
# written once. mean and rstd are stored to per-row scratch buffers for the
# backward pass.
#
# UNVERIFIED_SCALAR_STORE: gl.store of a scalar (mean, rstd) at a scalar
# pointer offset. Gluon's scalar-store behavior needs a GPU run to confirm.
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
    xc = x - mean  # centered; padded lanes hold x[pad] - mean, not zero yet
    # Mask xc so padded lanes do not contribute to variance.
    xc = gl.where(mask, xc, 0.0)
    var = gl.reduce(xc * xc, axis=0, combine_fn=_add_fn) / n_cols
    rstd = 1.0 / gl.sqrt(var + eps)
    w = gl.load(w_ptr + cols, mask=mask, other=0.0).to(gl.float32)
    b = gl.load(b_ptr + cols, mask=mask, other=0.0).to(gl.float32)
    y = xc * rstd * w + b
    gl.store(mean_ptr + row, mean)  # UNVERIFIED_SCALAR_STORE
    gl.store(rstd_ptr + row, rstd)  # UNVERIFIED_SCALAR_STORE
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
    gl.store(rstd_ptr + row, rstd)  # UNVERIFIED_SCALAR_STORE
    gl.store(y_ptr + row * row_stride + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Backward dx kernels
# ---------------------------------------------------------------------------

# LayerNorm backward dx kernel.
#
# Math (mirrors the Triton twin exactly):
#   x_hat = (x - mean) * rstd          (normalized input)
#   wdy   = w * dy                      (upstream grad scaled by weight)
#   c1    = sum(x_hat * wdy) / n_cols   (row scalar)
#   c2    = sum(wdy) / n_cols           (row scalar)
#   dx    = (wdy - x_hat * c1 - c2) * rstd
#
# UNVERIFIED_SCALAR_LOAD: gl.load(mean_ptr + row) / gl.load(rstd_ptr + row)
# return a scalar; whether Gluon requires an explicit view/unsqueeze for the
# subsequent broadcast arithmetic is unverified without a GPU run.
@gluon.jit
def _layernorm_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
                             row_stride, n_cols,
                             BLOCK: gl.constexpr, layout: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    dy = gl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
    x = gl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
    w = gl.load(w_ptr + cols, mask=mask, other=0.0).to(gl.float32)
    mean = gl.load(mean_ptr + row)   # UNVERIFIED_SCALAR_LOAD
    rstd = gl.load(rstd_ptr + row)   # UNVERIFIED_SCALAR_LOAD
    x_hat = gl.where(mask, (x - mean) * rstd, 0.0)
    wdy = gl.where(mask, w * dy, 0.0)
    c1 = gl.reduce(x_hat * wdy, axis=0, combine_fn=_add_fn) / n_cols
    c2 = gl.reduce(wdy, axis=0, combine_fn=_add_fn) / n_cols
    dx = (wdy - x_hat * c1 - c2) * rstd
    gl.store(dx_ptr + row * row_stride + cols, dx, mask=mask)


# RMSNorm backward dx kernel.
#
# Math (mirrors the Triton twin exactly):
#   x_hat = x * rstd                    (normalized input, no mean)
#   wdy   = w * dy
#   c1    = sum(x_hat * wdy) / n_cols
#   dx    = (wdy - x_hat * c1) * rstd
@gluon.jit
def _rmsnorm_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
                           row_stride, n_cols,
                           BLOCK: gl.constexpr, layout: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    dy = gl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
    x = gl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
    w = gl.load(w_ptr + cols, mask=mask, other=0.0).to(gl.float32)
    rstd = gl.load(rstd_ptr + row)   # UNVERIFIED_SCALAR_LOAD
    x_hat = gl.where(mask, x * rstd, 0.0)
    wdy = gl.where(mask, w * dy, 0.0)
    c1 = gl.reduce(x_hat * wdy, axis=0, combine_fn=_add_fn) / n_cols
    dx = (wdy - x_hat * c1) * rstd
    gl.store(dx_ptr + row * row_stride + cols, dx, mask=mask)


# ---------------------------------------------------------------------------
# Weight-gradient kernels (two-stage partial, no atomics)
# ---------------------------------------------------------------------------

# LayerNorm dw/db partial kernel -- Stage 1.
#
# Each of GROUP_M programs owns partial row g and accumulates every
# GROUP_M-th input row into it. Strided loop: row = g; row += GROUP_M.
# No atomics: rows never collide between programs.
#
# UNVERIFIED_WHILE_LOOP: dynamic while loop inside @gluon.jit -- mirrors
# the Triton twin's while-row-loop; Gluon's lowering of this pattern is
# unverified without a GPU run.
#
# UNVERIFIED_SCALAR_LOAD: gl.load(mean_ptr + row) inside the loop.
#
# UNVERIFIED_2D_PARTIAL_STORE: gl.store(pdw_ptr + g * n_cols + cols, ...) --
# linear-indexed [group, n_cols] buffer; unverified without a GPU run.
@gluon.jit
def _ln_dw_partial_kernel(dy_ptr, x_ptr, mean_ptr, rstd_ptr, pdw_ptr, pdb_ptr,
                          row_stride, n_rows, n_cols,
                          GROUP_M: gl.constexpr,
                          BLOCK: gl.constexpr, layout: gl.constexpr):
    g = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    acc_w = gl.zeros([BLOCK], dtype=gl.float32, layout=layout)
    acc_b = gl.zeros([BLOCK], dtype=gl.float32, layout=layout)
    row = g
    # UNVERIFIED_WHILE_LOOP
    while row < n_rows:
        dy = gl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
        x = gl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
        mean = gl.load(mean_ptr + row)   # UNVERIFIED_SCALAR_LOAD
        rstd = gl.load(rstd_ptr + row)   # UNVERIFIED_SCALAR_LOAD
        x_hat = (x - mean) * rstd
        acc_w = acc_w + dy * x_hat
        acc_b = acc_b + dy
        row = row + GROUP_M
    # UNVERIFIED_2D_PARTIAL_STORE
    gl.store(pdw_ptr + g * n_cols + cols, acc_w, mask=mask)
    gl.store(pdb_ptr + g * n_cols + cols, acc_b, mask=mask)


# RMSNorm dw partial kernel -- Stage 1.
#
# Same structure as LN partial but without mean; accumulates dy * (x * rstd).
@gluon.jit
def _rms_dw_partial_kernel(dy_ptr, x_ptr, rstd_ptr, pdw_ptr,
                           row_stride, n_rows, n_cols,
                           GROUP_M: gl.constexpr,
                           BLOCK: gl.constexpr, layout: gl.constexpr):
    g = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    acc_w = gl.zeros([BLOCK], dtype=gl.float32, layout=layout)
    row = g
    # UNVERIFIED_WHILE_LOOP
    while row < n_rows:
        dy = gl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
        x = gl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(gl.float32)
        rstd = gl.load(rstd_ptr + row)   # UNVERIFIED_SCALAR_LOAD
        acc_w = acc_w + dy * (x * rstd)
        row = row + GROUP_M
    # UNVERIFIED_2D_PARTIAL_STORE
    gl.store(pdw_ptr + g * n_cols + cols, acc_w, mask=mask)


# Reduction kernel -- Stage 2.
#
# Sums a [group_m, n_cols] partial buffer down to [n_cols].
# One program per column tile of BLOCK columns.
#
# UNVERIFIED_REDUCE_FOR_LOOP: for g in range(group_m) inside @gluon.jit --
# mirrors Triton's for g in range(group_m); Gluon may require a
# gl.constexpr upper bound (group_m is a runtime value here). Flagged.
@gluon.jit
def _dw_reduce_kernel(partial_ptr, out_ptr, group_m, n_cols,
                      BLOCK: gl.constexpr, layout: gl.constexpr):
    col_base = gl.program_id(0) * BLOCK
    cols = col_base + gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    acc = gl.zeros([BLOCK], dtype=gl.float32, layout=layout)
    # UNVERIFIED_REDUCE_FOR_LOOP: dynamic upper bound
    for g in range(group_m):
        acc = acc + gl.load(partial_ptr + g * n_cols + cols, mask=mask, other=0.0)
    gl.store(out_ptr + cols, acc, mask=mask)


# ---------------------------------------------------------------------------
# Host helpers for the two-stage weight gradient
# ---------------------------------------------------------------------------

def _reduce_partials(partial, n_cols):
    """Sum a [group, n_cols] float32 partial buffer to [n_cols]."""
    out = torch.empty(n_cols, device=partial.device, dtype=torch.float32)
    block = 256
    num_warps = 4
    size_per_thread = max(min(block // (32 * num_warps), 16 // partial.element_size()), 1)
    reduce_layout = gl.BlockedLayout(
        size_per_thread=[size_per_thread],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    grid = (triton.cdiv(n_cols, block),)
    _dw_reduce_kernel[grid](
        partial, out, partial.shape[0], n_cols,
        BLOCK=block, layout=reduce_layout, num_warps=num_warps,
    )
    return out


def _ln_dw_partial(dy, x, mean, rstd, dw, db, block, num_warps, layout):
    """Two-stage LayerNorm weight gradients (no atomics)."""
    n_rows, n_cols = x.shape
    group = min(_GROUP_M, n_rows)
    pdw = torch.zeros(group, n_cols, device=x.device, dtype=torch.float32)
    pdb = torch.zeros(group, n_cols, device=x.device, dtype=torch.float32)
    _ln_dw_partial_kernel[(group,)](
        dy, x, mean, rstd, pdw, pdb,
        x.stride(0), n_rows, n_cols,
        GROUP_M=group, BLOCK=block, layout=layout, num_warps=num_warps,
    )
    dw.copy_(_reduce_partials(pdw, n_cols))
    db.copy_(_reduce_partials(pdb, n_cols))


def _rms_dw_partial(dy, x, rstd, dw, block, num_warps, layout):
    """Two-stage RMSNorm weight gradient (no atomics)."""
    n_rows, n_cols = x.shape
    group = min(_GROUP_M, n_rows)
    pdw = torch.zeros(group, n_cols, device=x.device, dtype=torch.float32)
    _rms_dw_partial_kernel[(group,)](
        dy, x, rstd, pdw,
        x.stride(0), n_rows, n_cols,
        GROUP_M=group, BLOCK=block, layout=layout, num_warps=num_warps,
    )
    dw.copy_(_reduce_partials(pdw, n_cols))


# ---------------------------------------------------------------------------
# torch.autograd.Function wrappers
# ---------------------------------------------------------------------------

class _LayerNorm(torch.autograd.Function):
    """torch.autograd.Function wrapping the Gluon LayerNorm kernels."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        check_normalization_inputs(x, weight, bias)
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
        mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        _layernorm_fwd_kernel[(n_rows,)](
            x, weight, bias, y, mean, rstd,
            x.stride(0), n_cols, eps,
            BLOCK=block, layout=layout, num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        ctx.block = block
        ctx.num_warps = num_warps
        ctx.layout = layout
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        n_rows, n_cols = x.shape
        block = ctx.block
        num_warps = ctx.num_warps
        layout = ctx.layout
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        _layernorm_bwd_dx_kernel[(n_rows,)](
            dy, x, weight, mean, rstd, dx,
            x.stride(0), n_cols,
            BLOCK=block, layout=layout, num_warps=num_warps,
        )
        dw = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
        db = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
        _ln_dw_partial(dy, x, mean, rstd, dw, db, block, num_warps, layout)
        return dx, dw.to(weight.dtype), db.to(weight.dtype), None


class _RMSNorm(torch.autograd.Function):
    """torch.autograd.Function wrapping the Gluon RMSNorm kernels."""

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
        ctx.block = block
        ctx.num_warps = num_warps
        ctx.layout = layout
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x.shape
        block = ctx.block
        num_warps = ctx.num_warps
        layout = ctx.layout
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        _rmsnorm_bwd_dx_kernel[(n_rows,)](
            dy, x, weight, rstd, dx,
            x.stride(0), n_cols,
            BLOCK=block, layout=layout, num_warps=num_warps,
        )
        dw = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
        _rms_dw_partial(dy, x, rstd, dw, block, num_warps, layout)
        return dx, dw.to(weight.dtype), None


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim of a 2-D CUDA tensor (differentiable).

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
    """RMSNorm over the last dim of a 2-D CUDA tensor (differentiable).

    Args:
        x: 2-D, contiguous CUDA tensor (float16/bfloat16/float32/float64).
        weight: 1-D CUDA tensor of length x.shape[1], same dtype as x.
        eps: Numerical stability epsilon. Defaults to 1e-5.

    Returns:
        Normalized tensor, same shape and dtype as x.
    """
    return _RMSNorm.apply(x, weight, eps)
