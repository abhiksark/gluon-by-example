# src/gluon_by_example/_validation.py
"""Input validation shared by all kernels."""

import torch

# Dtypes verified against torch.softmax. fp8 is excluded: Triton cannot
# compile fp8e4nv on consumer Ampere, and torch.softmax refuses fp8 outright.
_SOFTMAX_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def check_elementwise_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    """Validates inputs for elementwise binary kernels.

    Args:
        x: First input tensor.
        y: Second input tensor.

    Raises:
        ValueError: If inputs are not CUDA tensors, or differ in shape or
            dtype, or are not contiguous.
    """
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("inputs must be CUDA tensors")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} vs {y.dtype}")
    if not (x.is_contiguous() and y.is_contiguous()):
        raise ValueError("inputs must be contiguous")


def check_softmax_inputs(x: torch.Tensor) -> None:
    """Validates input for row-wise softmax kernels.

    Args:
        x: Input tensor.

    Raises:
        ValueError: If the input is not a 2-D, contiguous CUDA tensor of a
            supported floating dtype.
    """
    if not x.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if x.ndim != 2:
        raise ValueError(f"input must be 2-D, got {x.ndim}-D")
    if x.dtype not in _SOFTMAX_DTYPES:
        supported = ", ".join(str(d).removeprefix("torch.") for d in _SOFTMAX_DTYPES)
        raise ValueError(
            f"unsupported dtype {x.dtype}; supported floating dtypes: {supported}"
        )
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")
