# src/gluon_by_example/triton_impl/scan.py
"""Row-wise inclusive prefix sum (cumsum) via tl.associative_scan.

Scan is the third parallel primitive after map and reduce. Unlike a reduction,
each output depends on all earlier elements (a carry dependency); associative
scan breaks that into a parallel prefix instead of a sequential loop.
"""

import torch
import triton
import triton.language as tl

from gluon_by_example._validation import check_softmax_inputs

_MAX_COLS = 32768


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def _cumsum_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.associative_scan(x, axis=0, combine_fn=_add)
    tl.store(out_ptr + row * row_stride + cols, y, mask=mask)


def cumsum(x: torch.Tensor) -> torch.Tensor:
    """Row-wise inclusive prefix sum of a 2-D contiguous CUDA tensor."""
    check_softmax_inputs(x)  # same 2-D contiguous CUDA float contract
    n_rows, n_cols = x.shape
    if n_cols == 0:
        raise ValueError("rows must be non-empty")
    if n_cols > _MAX_COLS:
        raise ValueError(f"n_cols={n_cols} exceeds fused-row limit {_MAX_COLS}")
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    num_warps = 8 if block >= 2048 else 4
    _cumsum_kernel[(n_rows,)](x, out, x.stride(0), n_cols, BLOCK=block, num_warps=num_warps)
    return out
