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


# Tensor-core dtypes for attention; head_dim must be a power of two the
# kernel can tile as a constexpr.
_ATTENTION_DTYPES = (torch.float16, torch.bfloat16)
_ATTENTION_HEAD_DIMS = (16, 32, 64, 128)


def check_attention_inputs(q: torch.Tensor, k: torch.Tensor,
                           v: torch.Tensor) -> None:
    """Validates inputs for 4-D (Z, H, N, D) attention kernels.

    Args:
        q: Queries, shape (Z, H, N, D).
        k: Keys, same shape and dtype as q.
        v: Values, same shape and dtype as q.

    Raises:
        ValueError: If the tensors are not 4-D, contiguous, equal-shaped CUDA
            tensors of a matching tensor-core dtype with a supported head_dim.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.ndim != 4:
            raise ValueError(f"{name} must be 4-D (Z, H, N, D), got {t.ndim}-D")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"shape mismatch: {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"dtype mismatch: {q.dtype} {k.dtype} {v.dtype}"
        )
    if q.dtype not in _ATTENTION_DTYPES:
        supported = ", ".join(str(d).removeprefix("torch.") for d in _ATTENTION_DTYPES)
        raise ValueError(f"unsupported dtype {q.dtype}; supported: {supported}")
    if q.shape[-1] not in _ATTENTION_HEAD_DIMS:
        raise ValueError(
            f"head_dim {q.shape[-1]} unsupported; must be one of {_ATTENTION_HEAD_DIMS}"
        )
