# src/gluon_by_example/gluon_impl/softmax.py
"""Row-wise fused softmax in Gluon."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gluon_by_example._validation import check_softmax_inputs

# Same fused-row strategy and cap as the Triton twin: one program holds one
# full row in registers.
_MAX_COLS = 32768


# gl.max / gl.sum sugar exists; we spell out reduce + combine_fn to show
# what a reduction is made of.
@gluon.jit
def _max_fn(a, b):
    return gl.maximum(a, b)


@gluon.jit
def _add_fn(a, b):
    return a + b


@gluon.jit
def _softmax_kernel(x_ptr, out_ptr, row_stride, n_cols,
                    BLOCK: gl.constexpr, layout: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    mask = cols < n_cols
    x = gl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(gl.float32)
    x = x - gl.reduce(x, axis=0, combine_fn=_max_fn)
    numerator = gl.exp(x)
    denominator = gl.reduce(numerator, axis=0, combine_fn=_add_fn)
    # gl.store casts the fp32 result back to the output dtype.
    gl.store(out_ptr + row * row_stride + cols, numerator / denominator, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Computes row-wise softmax of a 2-D CUDA tensor in one fused Gluon kernel.

    Same algorithm as the Triton twin: each row is loaded once, normalized
    on-chip in float32, and written once. float64 inputs are accumulated in
    float32 too; expect fp32-level precision.

    Args:
        x: 2-D, contiguous CUDA tensor (float16/bfloat16/float32/float64)
            with at least 1 and at most 32768 columns.

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
    # A hand-chosen layout: each thread owns one contiguous run of the row.
    # The run is capped at 16 bytes so consecutive lanes stay 16B apart and
    # warp loads coalesce — uncapped runs (e.g. 128B at BLOCK=16384) scatter
    # each warp access across 32 cache lines and cost ~11% at wide rows.
    # This is the same dtype-width logic Triton's layout inference applies.
    # For rows narrower than the CTA's 32 * num_warps lanes the layout is
    # larger than the tensor and Gluon replicates elements across lanes.
    size_per_thread = max(min(block // (32 * num_warps), 16 // x.element_size()), 1)
    layout = gl.BlockedLayout(
        size_per_thread=[size_per_thread],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    _softmax_kernel[(n_rows,)](
        x, out, x.stride(0), n_cols, BLOCK=block, layout=layout, num_warps=num_warps
    )
    return out
