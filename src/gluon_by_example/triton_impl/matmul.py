# src/gluon_by_example/triton_impl/matmul.py
"""Tiled, autotuned matmul in standard Triton."""

import torch
import triton
import triton.language as tl

from gluon_by_example._validation import check_matmul_inputs

# A small Ampere-flavored search space: tile shapes, pipeline depth, and warp
# count. Chapters 1-3 hardcoded launch configs; matmul is where tuning starts
# paying for itself. The first config sits at 96KB of the 99KB CC 8.6 shared
# memory budget (the pipeliner buffers num_stages - 1 tiles); on cards with
# less shared memory the autotuner silently scores failing configs as inf
# and skips them, so the list degrades gracefully rather than crashing.
_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
        num_stages=3, num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=4, num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=5, num_warps=2,
    ),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # Grouped ordering: walk C in GROUP_M-tall column-major super-rows so
    # neighboring programs reuse the same A rows and B columns through L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row/col indices wrap with % so out-of-range lanes load valid (unused)
    # data; the store mask below is what keeps C correct.
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Accumulate in fp32, downcast once at the end.
    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes a @ b for 2-D tensor-core dtypes (float16/bfloat16).

    Tiled with a grouped, L2-friendly program ordering; accumulates in
    float32. The launch config is autotuned: the first call for a given
    (M, N, K) tries 6 candidate configs, then the winner is cached.

    Args:
        a: (M, K) CUDA tensor, float16 or bfloat16, contiguous.
        b: (K, N) CUDA tensor, same dtype as a, contiguous.

    Returns:
        (M, N) tensor of the same dtype holding a @ b.
    """
    check_matmul_inputs(a, b)
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    _matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
