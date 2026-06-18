# src/gluon_by_example/gluon_impl/matmul.py
"""Tiled matmul in Gluon: explicit layouts + mma_v2, no async pipeline.

This is the floor kernel from the matmul-5090 experiment: plain gl.load tiles
feeding mma_v2 with fp32 accumulation. It is the honest best-Gluon result: on
the 5090 it reached 0.80x of Triton, and adding cp.async/TMA pipelining (see the
article) made it slower, not faster. The chapter's point is the layout/mma
mechanics and that verdict, not a production gemm, so the contract is
deliberately narrow: square-tile-aligned float16 only.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from gluon_by_example._validation import check_matmul_inputs

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 64
_NUM_WARPS = 8


@gluon.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                   BLOCK_K: gl.constexpr, acc_layout: gl.constexpr,
                   lhs_layout: gl.constexpr, rhs_layout: gl.constexpr,
                   load_layout: gl.constexpr):
    # One program computes one BLOCK_M x BLOCK_N output tile. No masking: the
    # wrapper guarantees tile-aligned shapes, so every load is in-bounds.
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)

    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_layout))
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, load_layout))

    acc = gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout)
    for k0 in range(0, K, BLOCK_K):
        offs_ka = k0 + gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_layout))
        offs_kb = k0 + gl.arange(0, BLOCK_K, gl.SliceLayout(1, load_layout))
        a_tile = gl.load(a_ptr + offs_m[:, None] * K + offs_ka[None, :])
        b_tile = gl.load(b_ptr + offs_kb[:, None] * N + offs_n[None, :])
        acc = mma_v2(gl.convert_layout(a_tile, lhs_layout),
                     gl.convert_layout(b_tile, rhs_layout),
                     acc)

    out = gl.convert_layout(acc.to(gl.float16), load_layout)
    gl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], out)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes a @ b in Gluon for tile-aligned float16 inputs.

    Contract (narrow on purpose, see module docstring): a is (M, K) and b is
    (K, N), both float16, CUDA, contiguous, with M and N divisible by 128 and K
    divisible by 64. Accumulates in float32, returns float16.

    Args:
        a: (M, K) CUDA tensor, float16, contiguous, M % 128 == 0, K % 64 == 0.
        b: (K, N) CUDA tensor, float16, contiguous, N % 128 == 0.

    Returns:
        (M, N) float16 tensor holding a @ b.

    Raises:
        ValueError: on non-CUDA, dtype != float16, inner-dim mismatch, or
            shapes not divisible by the (128, 128, 64) tile.
    """
    check_matmul_inputs(a, b)
    if a.dtype != torch.float16:
        raise ValueError(
            f"gluon matmul supports float16 only, got {a.dtype}; this chapter "
            "kernel is fp16 + tile-aligned (see chapters/05-matmul-gluon)")
    M, K = a.shape
    _, N = b.shape
    if M % _BLOCK_M or N % _BLOCK_N or K % _BLOCK_K:
        raise ValueError(
            f"gluon matmul needs tile-aligned shapes: M % {_BLOCK_M}, "
            f"N % {_BLOCK_N}, K % {_BLOCK_K} must be 0, got M={M} N={N} K={K}")

    # Layouts are built here (where num_warps is a plain int) and passed in as
    # constexpr params, exactly the pattern the committed gluon softmax kernel
    # uses. num_warps stays a reserved launch kwarg.
    acc_layout = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_NUM_WARPS, 1], instr_shape=[16, 8])
    lhs_layout = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=8)
    rhs_layout = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=8)
    load_layout = gl.BlockedLayout([1, 8], [4, 8], [_NUM_WARPS, 1], [1, 0])

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = (M // _BLOCK_M, N // _BLOCK_N)
    _matmul_kernel[grid](a, b, c, M, N, K, _BLOCK_M, _BLOCK_N, _BLOCK_K,
                         acc_layout, lhs_layout, rhs_layout, load_layout,
                         num_warps=_NUM_WARPS)
    return c
