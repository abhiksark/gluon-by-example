# src/gluon_by_example/gluon_impl/vector_add.py
"""Vector addition in Gluon.

Same algorithm as the Triton version, but the register layout that Triton
infers automatically is declared explicitly here. This is the core Gluon
trade: more verbosity, full control over data placement.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from gluon_by_example._validation import check_elementwise_inputs

_BLOCK = 1024
# 8 elements per thread x 32 threads per warp x 4 warps = 1024 elements per CTA.
_LAYOUT = gl.BlockedLayout(
    size_per_thread=[8],
    threads_per_warp=[32],
    warps_per_cta=[4],
    order=[0],
)


@gluon.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    offsets = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offsets < n
    x = gl.load(x_ptr + offsets, mask=mask)
    y = gl.load(y_ptr + offsets, mask=mask)
    gl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes x + y elementwise on contiguous CUDA tensors of equal shape.

    Args:
        x: CUDA tensor.
        y: CUDA tensor with the same shape and dtype as x.

    Returns:
        New tensor holding x + y.
    """
    check_elementwise_inputs(x, y)
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, _BLOCK),)
    _add_kernel[grid](x, y, out, n, BLOCK=_BLOCK, layout=_LAYOUT)
    return out
