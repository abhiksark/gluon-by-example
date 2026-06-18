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


# Tensor-core dtypes verified against torch.matmul. fp32 is excluded on
# purpose: tl.dot defaults to tf32 while torch defaults to ieee — comparing
# them honestly is its own chapter-sized story.
_MATMUL_DTYPES = (torch.float16, torch.bfloat16)


def check_matmul_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    """Validates inputs for 2-D matmul kernels.

    Args:
        a: Left operand, shape (M, K).
        b: Right operand, shape (K, N).

    Raises:
        ValueError: If the inputs are not 2-D, contiguous CUDA tensors of a
            matching supported tensor-core dtype with compatible, non-empty
            shapes.
    """
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("inputs must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"inputs must be 2-D, got {a.ndim}-D and {b.ndim}-D")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
    if a.dtype not in _MATMUL_DTYPES:
        supported = ", ".join(str(d).removeprefix("torch.") for d in _MATMUL_DTYPES)
        raise ValueError(
            f"unsupported dtype {a.dtype}; supported tensor-core dtypes: {supported}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions mismatch: {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if min(a.shape) == 0 or min(b.shape) == 0:
        raise ValueError("dimensions must be non-empty")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("inputs must be contiguous")


# Same fused-row dtypes as softmax. fp64 included so gradcheck can run.
_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def check_normalization_inputs(x: torch.Tensor, weight: torch.Tensor,
                               bias: torch.Tensor = None) -> None:
    """Validates inputs for row-wise LayerNorm/RMSNorm kernels.

    Args:
        x: Input tensor, shape (M, N).
        weight: Per-feature scale, shape (N,).
        bias: Optional per-feature shift, shape (N,).

    Raises:
        ValueError: If x is not a 2-D, contiguous CUDA tensor of a supported
            floating dtype, or if weight/bias are not 1-D CUDA tensors of
            length N and matching dtype.
    """
    if not x.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if x.ndim != 2:
        raise ValueError(f"input must be 2-D, got {x.ndim}-D")
    if x.dtype not in _NORM_DTYPES:
        supported = ", ".join(str(d).removeprefix("torch.") for d in _NORM_DTYPES)
        raise ValueError(f"unsupported dtype {x.dtype}; supported: {supported}")
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")
    n_cols = x.shape[1]
    for name, t in (("weight", weight), ("bias", bias)):
        if t is None:
            continue
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.ndim != 1 or t.shape[0] != n_cols:
            raise ValueError(
                f"{name} must be 1-D of length {n_cols}, got {tuple(t.shape)}"
            )
        if t.dtype != x.dtype:
            raise ValueError(f"{name} dtype {t.dtype} must match input {x.dtype}")
