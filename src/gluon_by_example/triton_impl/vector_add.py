# src/gluon_by_example/triton_impl/vector_add.py
"""Vector addition in standard Triton."""

import torch
import triton
import triton.language as tl

from gluon_by_example._validation import check_elementwise_inputs

_BLOCK = 1024


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


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
    _add_kernel[grid](x, y, out, n, BLOCK=_BLOCK)
    return out
