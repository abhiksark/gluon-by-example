# src/gluon_by_example/gluon_impl/matmul_pipelined.py
"""cp.async multi-stage pipelined fp16 matmul in Gluon: the floor + a prefetch ring.

This is the floor kernel (explicit layouts + mma_v2, fp32 accumulate) wrapped in
an explicit cp.async pipeline plus GROUP_M L2-aware tile ordering. A STAGES-deep
shared-memory ring buffer prefetches each K-tile's global->shared copies
(STAGES-1) iterations ahead, so the next tile's loads overlap the current tile's
mma_v2 instead of stalling on them; the GROUP_M grouped raster (1-D grid) makes
neighboring programs reuse A/B through L2. At the Triton-matched footprint
(BLOCK_K=32, STAGES=3, num_warps=4) it beats the unpipelined floor at every size,
but a codegen gap keeps it at ~0.75x of the Triton compiler's autotuned GEMM. The
full story — footprint vs occupancy, the GROUP_M fix, and why the last ~25% is a
software-pipeliner pass Gluon does not run — is in chapter 5.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2, async_copy

from gluon_by_example._validation import check_matmul_inputs

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 32
_STAGES = 3
_NUM_WARPS = 4
_GROUP_M = 8


@gluon.jit
def _pipe_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                        BLOCK_K: gl.constexpr, STAGES: gl.constexpr,
                        GROUP_M: gl.constexpr,
                        acc_layout: gl.constexpr, lhs_layout: gl.constexpr,
                        rhs_layout: gl.constexpr, load_layout: gl.constexpr,
                        a_smem_layout: gl.constexpr, b_smem_layout: gl.constexpr):
    # One program computes one BLOCK_M x BLOCK_N output tile. No masking: the
    # wrapper guarantees tile-aligned shapes, so every copy/store is in-bounds.
    # Grouped (GROUP_M) raster: walk C in GROUP_M-tall super-rows from a 1-D
    # grid so neighboring programs reuse A rows / B columns through L2 — the same
    # L2-aware ordering the Triton twin uses. Without it the plain column-major
    # walk thrashes L2 once the matrices outgrow it (the gap widens at large N).
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_layout))
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, load_layout))
    # K offsets for the async-copy address tensors (one tile wide).
    offs_ka = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_layout))
    offs_kb = gl.arange(0, BLOCK_K, gl.SliceLayout(1, load_layout))

    # Ring buffers: STAGES deep. A and B carry separate smem layouts because the
    # NVMMA swizzle width is capped by the inner contiguous dim: at BLOCK_K=32
    # the A tile inner dim is 32 elems (64B), so A uses a 64B swizzle while B
    # (inner dim BLOCK_N=128) keeps the wider 128B swizzle.
    a_smem = gl.allocate_shared_memory(gl.float16, [STAGES, BLOCK_M, BLOCK_K], a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float16, [STAGES, BLOCK_K, BLOCK_N], b_smem_layout)

    k_tiles = K // BLOCK_K

    # ----- prologue: prefetch the first (STAGES-1) tiles, one group each -----
    for s in range(STAGES - 1):
        if s < k_tiles:
            koff = s * BLOCK_K
            a_ptrs = a_ptr + offs_m[:, None] * K + (koff + offs_ka)[None, :]
            b_ptrs = b_ptr + (koff + offs_kb)[:, None] * N + offs_n[None, :]
            async_copy.async_copy_global_to_shared(a_smem.index(s), a_ptrs)
            async_copy.async_copy_global_to_shared(b_smem.index(s), b_ptrs)
        async_copy.commit_group()

    acc = gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout)

    for k in range(k_tiles):
        # Issue the copy for the tile (STAGES-1) ahead, into its ring slot.
        nxt = k + (STAGES - 1)
        if nxt < k_tiles:
            slot = nxt % STAGES
            koff = nxt * BLOCK_K
            a_ptrs = a_ptr + offs_m[:, None] * K + (koff + offs_ka)[None, :]
            b_ptrs = b_ptr + (koff + offs_kb)[:, None] * N + offs_n[None, :]
            async_copy.async_copy_global_to_shared(a_smem.index(slot), a_ptrs)
            async_copy.async_copy_global_to_shared(b_smem.index(slot), b_ptrs)
        async_copy.commit_group()

        # Wait until tile k's group is done. After issuing the group above there
        # are (STAGES-1) groups for tiles k..k+STAGES-1 still relevant; we need
        # tile k complete, so allow (STAGES-2) groups to remain outstanding.
        # (Extra trailing empty commit_groups in the tail only ever lower the
        # outstanding count, so this stays safe near the end.)
        async_copy.wait_group(STAGES - 2)

        # Direct smem->reg: load straight into the DotOperandLayout the mma
        # consumes, no intermediate blocked load + convert_layout.
        cur = k % STAGES
        a = a_smem.index(cur).load(lhs_layout)
        b = b_smem.index(cur).load(rhs_layout)
        acc = mma_v2(a, b, acc)

    out = gl.convert_layout(acc.to(gl.float16), load_layout)
    gl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], out)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes a @ b with a cp.async-pipelined Gluon kernel (fp16, tile-aligned).

    Same narrow contract as the floor kernel: a is (M, K) and b is (K, N), both
    float16, CUDA, contiguous, with M and N divisible by 128 and K divisible by
    32 (the pipelined kernel uses BLOCK_K=32). Accumulates in float32, returns
    float16.

    Args:
        a: (M, K) CUDA tensor, float16, contiguous, M % 128 == 0, K % 32 == 0.
        b: (K, N) CUDA tensor, float16, contiguous, N % 128 == 0.

    Returns:
        (M, N) float16 tensor holding a @ b.

    Raises:
        ValueError: on non-CUDA, dtype != float16, inner-dim mismatch, or
            shapes not divisible by the (128, 128, 32) tile.
    """
    check_matmul_inputs(a, b)
    if a.dtype != torch.float16:
        raise ValueError(
            f"gluon pipelined matmul supports float16 only, got {a.dtype}; this "
            "chapter kernel is fp16 + tile-aligned (see chapters/05-matmul-gluon)")
    M, K = a.shape
    _, N = b.shape
    if M % _BLOCK_M or N % _BLOCK_N or K % _BLOCK_K:
        raise ValueError(
            f"gluon pipelined matmul needs tile-aligned shapes: M % {_BLOCK_M}, "
            f"N % {_BLOCK_N}, K % {_BLOCK_K} must be 0, got M={M} N={N} K={K}")

    # Layouts are built here (where num_warps is a plain int) and passed in as
    # constexpr params, mirroring the floor wrapper. acc spreads the MMA across
    # num_warps warps along M; the dot-operand layouts inherit it; the blocked
    # load_layout tiles the same BLOCK_M for the fp16 store.
    acc_layout = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_NUM_WARPS, 1], instr_shape=[16, 8])
    lhs_layout = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=8)
    rhs_layout = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=8)
    load_layout = gl.BlockedLayout([1, 8], [4, 8], [_NUM_WARPS, 1], [1, 0])

    # NVMMA swizzle width is capped by the inner contiguous dim (fp16 -> a B-byte
    # swizzle needs inner_dim >= B/2 elems). A inner = BLOCK_K = 32 -> 64B fits
    # (128B would need 64); B inner = BLOCK_N = 128 -> 128B fits.
    a_smem_layout = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=16, rank=2)
    b_smem_layout = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = ((M // _BLOCK_M) * (N // _BLOCK_N),)
    _pipe_matmul_kernel[grid](
        a, b, c, M, N, K, _BLOCK_M, _BLOCK_N, _BLOCK_K, _STAGES, _GROUP_M,
        acc_layout, lhs_layout, rhs_layout, load_layout,
        a_smem_layout, b_smem_layout, num_warps=_NUM_WARPS)
    return c
