# src/gluon_by_example/triton_impl/softmax.py
"""Row-wise fused softmax in standard Triton."""

import torch
import triton
import triton.language as tl

from gluon_by_example._validation import check_softmax_inputs

# One program handles one full row held in registers, so rows are capped at
# what compiles sensibly. Beyond this, a multi-pass kernel is the right tool.
_MAX_COLS = 32768


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    # tl.store casts the fp32 result back to the output dtype.
    tl.store(out_ptr + row * row_stride + cols, numerator / denominator, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Computes row-wise softmax of a 2-D CUDA tensor in one fused kernel.

    Each row is loaded once, normalized on-chip in float32, and written once —
    versus the ~8 memory passes of unfused eager softmax.

    Args:
        x: 2-D, floating-point, contiguous CUDA tensor with at least 1 and
            at most 32768 columns.

    Returns:
        New tensor of the same shape and dtype with softmax applied along
        the last dimension.
    """
    check_softmax_inputs(x)
    n_rows, n_cols = x.shape
    if n_cols == 0:
        raise ValueError("rows must be non-empty")
    if n_cols > _MAX_COLS:
        raise ValueError(f"n_cols={n_cols} exceeds fused-row limit {_MAX_COLS}")
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    # Wider rows give the single-row reductions more work, so add warps.
    num_warps = 4
    if block >= 2048:
        num_warps = 8
    if block >= 8192:
        num_warps = 16
    _softmax_kernel[(n_rows,)](
        x, out, x.stride(0), n_cols, BLOCK=block, num_warps=num_warps
    )
    return out
