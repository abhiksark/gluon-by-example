# src/gluon_by_example/gluon_impl/attention.py
"""FlashAttention forward in Gluon (experimental).

Tiles over KV blocks with an online-softmax running rescale (FA2 algorithm)
and two tensor-core matmuls (QK^T, then P V) via mma_v2. Never materializes
the N-by-N score matrix. Forward saves the per-row logsumexp L; the backward
is deferred to P5 and raises NotImplementedError here.

Inputs are 4-D (Z, H, N, D); the wrapper views them as (Z*H, N, D), exactly
matching the Triton twin in triton_impl/attention.py.

STATUS: static-checked only, not GPU-run. See module-level CONCERNS list
for gl.* calls that cannot be confirmed without a live Gluon runtime.

CONCERNS (flagged, not silently guessed):
  1. gl.DotOperandLayout / gl.NVMMADistributedLayout for the QK^T output fed
     into P V: after mma_v2 produces QK (acc_layout = NVMMADistributedLayout),
     converting it to lhs_layout (DotOperandLayout operand_index=0) for the
     second mma_v2 call may require a different k_width because QK has shape
     (BLOCK_M, BLOCK_N) while the second matmul contracts over BLOCK_N. The
     k_width=BLOCK_N//8 guess below follows the pattern in matmul.py (k_width=8
     for BLOCK_K=64) but BLOCK_N=64 -> k_width=8 only by coincidence; a GPU run
     must confirm.
  2. acc * alpha[:, None] inside the kernel loop: Gluon's distributed MMA layout
     (NVMMADistributedLayout) may not support in-place broadcast-multiply with a
     [BLOCK_M] vector. The workaround proposed is gl.convert_layout(alpha, ...)
     followed by elementwise multiply, but the exact layout argument is not
     confirmed.
  3. gl.store of a scalar per-row logsumexp L to a 1-D slice: the pattern
     gl.store(L + off_b * N + offs_m, lse, mask=offs_m < N) matches the Triton
     twin textually but its behaviour under the blocked layout for a rank-1
     pointer arithmetic in Gluon is unverified.
  4. p.to(v.dtype) inside the kernel before the PV mma_v2: the in-kernel cast
     of the NVMMADistributed fp32 tile to fp16 before the second mma_v2 is
     idiomatic from the Triton twin but Gluon's .to() on distributed layouts may
     carry caveats not visible from static inspection.
  5. gl.where masking with a row_layout-derived condition applied to acc_layout
     operands: the padding mask (offs_n[None, :] < N) and causal mask
     (offs_m[:, None] >= offs_n[None, :]) are built from row_layout arange
     vectors, but the true/false branches are acc_layout (NVMMADistributedLayout)
     tensors. Whether gl.where accepts a condition whose layout differs from
     the operand layout, and how it broadcasts the 1-D row_layout index across
     the 2-D acc_layout tile, is unverified without a live Gluon runtime.
"""

import math

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from gluon_by_example._validation import check_attention_inputs

_BLOCK = 64   # BLOCK_M == BLOCK_N keeps causal block alignment exact, matches Triton twin
_NUM_WARPS = 4  # 64x64 attention tile; matmul.py uses 8 warps for its 128x128 tile


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@gluon.jit
def _attn_fwd_kernel(
        Q, K, V, O, L, sm_scale,  # noqa: E741
        stride_qb, stride_qm, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_ob, stride_om, stride_od,
        N,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, D: gl.constexpr,
        CAUSAL: gl.constexpr,
        acc_layout: gl.constexpr,
        lhs_layout: gl.constexpr,
        rhs_layout: gl.constexpr,
        load_layout: gl.constexpr,
        row_layout: gl.constexpr,
        col_layout: gl.constexpr):
    """Forward FA kernel: online-softmax tiling over KV blocks.

    One program (start_m, off_b) owns BLOCK_M query rows for one batch-head
    off_b. It iterates over KV blocks, maintaining running max m_i, running
    sum l_i, and accumulator acc in fp32. Two mma_v2 calls handle QK^T and
    PV respectively.

    Layout names follow matmul.py: acc_layout = NVMMADistributedLayout for
    fp32 accumulation; lhs/rhs = DotOperandLayout for operands; load_layout =
    BlockedLayout for input loads; row_layout = 1-D BlockedLayout for the
    per-row running statistics (m_i, l_i, alpha, lse).
    """
    start_m = gl.program_id(0)
    off_b = gl.program_id(1)

    # Row indices for this program's BLOCK_M tile.
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    # Head-dim indices.
    offs_d = gl.arange(0, D, layout=col_layout)

    # Load Q tile: (BLOCK_M, D) -- loaded once, reused for all KV blocks.
    # CONCERN 3 (mild): the strided pointer arithmetic below mirrors the Triton
    # twin; Gluon's BlockedLayout pointer indexing is assumed to behave
    # identically for contiguous tensors.
    q_tile = gl.load(
        Q + off_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < N,
        other=0.0,
    )

    # Running statistics (all fp32).
    m_i = gl.full([BLOCK_M], -float("inf"), gl.float32, row_layout)
    l_i = gl.full([BLOCK_M], 0.0, gl.float32, row_layout)
    acc = gl.full([BLOCK_M, D], 0.0, gl.float32, acc_layout)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + gl.arange(0, BLOCK_N, layout=row_layout)

        # Load K tile transposed: (D, BLOCK_N).
        k_tile = gl.load(
            K + off_b * stride_kb + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
            mask=offs_n[None, :] < N,
            other=0.0,
        )

        # QK^T: (BLOCK_M, BLOCK_N), accumulated in fp32.
        # Convert both operands to their DotOperandLayouts before mma_v2.
        qk = mma_v2(
            gl.convert_layout(q_tile, lhs_layout),
            gl.convert_layout(k_tile, rhs_layout),
            gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout),
        )
        # Scale and apply padding mask (out-of-bounds columns become -inf).
        qk = qk * sm_scale
        # CONCERN 5: gl.where mixes a condition derived from row_layout arange
        # vectors (offs_n[None, :] < N) with acc_layout (NVMMADistributedLayout)
        # operands.  Whether Gluon accepts a cross-layout condition and how it
        # broadcasts the 1-D row_layout index across the 2-D acc_layout tile is
        # unverified without a GPU run.
        qk_masked = gl.where(
            offs_n[None, :] < N,
            qk,
            gl.full([BLOCK_M, BLOCK_N], -float("inf"), gl.float32, acc_layout),
        )
        if CAUSAL:
            qk_masked = gl.where(
                offs_m[:, None] >= offs_n[None, :],
                qk_masked,
                gl.full([BLOCK_M, BLOCK_N], -float("inf"), gl.float32, acc_layout),
            )

        # Online-softmax update.
        m_ij = gl.maximum(m_i, gl.reduce(qk_masked, axis=1, combine_fn=_max_fn))
        p = gl.exp(qk_masked - m_ij[:, None])
        alpha = gl.exp(m_i - m_ij)

        # Rescale running sum and accumulator.
        # CONCERN 2: acc * alpha[:, None] -- broadcast-multiply of a
        # NVMMADistributedLayout tensor by a row_layout vector. Not confirmed;
        # the [:, None] broadcasting pattern mirrors Triton tl.float32 tiles.
        l_i = l_i * alpha + gl.reduce(p, axis=1, combine_fn=_add_fn)
        acc = acc * alpha[:, None]

        # Load V tile: (BLOCK_N, D).
        v_tile = gl.load(
            V + off_b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < N,
            other=0.0,
        )

        # PV: acc += p @ V, accumulated in fp32.
        # CONCERN 1: converting the QK output (acc_layout, BLOCK_M x BLOCK_N)
        # to lhs_layout2 (DotOperandLayout, operand_index=0) for the second
        # mma_v2. k_width here is BLOCK_N // 8 = 8 (for BLOCK_N=64), same
        # as in matmul.py. However, the actual k_width contract for the PV
        # matmul dimension (BLOCK_N) is unverified without a GPU run.
        # CONCERN 4: p.to(gl.float16) cast on a NVMMADistributedLayout tensor
        # before feeding into the second mma_v2.
        acc = mma_v2(
            gl.convert_layout(p.to(gl.float16), lhs_layout),
            gl.convert_layout(v_tile, rhs_layout),
            acc,
        )

        m_i = m_ij

    # Normalize.
    acc = acc / l_i[:, None]
    lse = m_i + gl.log(l_i)

    # Store output O: (BLOCK_M, D).
    # CONCERN 3: store under acc_layout with convert to output dtype.
    out = gl.convert_layout(acc.to(gl.float16), load_layout)
    gl.store(
        O + off_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        out,
        mask=offs_m[:, None] < N,
    )

    # Store per-row logsumexp L: shape (b, N), row-major.
    # CONCERN 3: scalar gl.store to 1-D pointer with row_layout -- assumes
    # Gluon handles this as a 1-D blocked store, matching Triton's tl.store.
    gl.store(L + off_b * N + offs_m, lse, mask=offs_m < N)


# ---------------------------------------------------------------------------
# combine_fn helpers (same pattern as softmax.py)
# ---------------------------------------------------------------------------

@gluon.jit
def _max_fn(a, b):
    return gl.maximum(a, b)


@gluon.jit
def _add_fn(a, b):
    return a + b


# ---------------------------------------------------------------------------
# Shape helper (matches triton twin exactly)
# ---------------------------------------------------------------------------

def _shape3(t):
    z, h, n, d = t.shape
    return t.reshape(z * h, n, d)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _Attention(torch.autograd.Function):
    """Autograd wrapper: forward launches the Gluon FA kernel; backward deferred."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        check_attention_inputs(q, k, v)
        z, h, n, d = q.shape
        sm_scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale

        q3, k3, v3 = _shape3(q), _shape3(k), _shape3(v)
        o3 = torch.empty_like(q3)
        b = z * h
        L = torch.empty((b, n), device=q.device, dtype=torch.float32)

        # Layout construction: same pattern as matmul.py.
        # acc_layout accumulates the (BLOCK_M, D) and (BLOCK_M, BLOCK_N) tiles
        # in fp32 via mma_v2.
        acc_layout = gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[_NUM_WARPS, 1],
            instr_shape=[16, 8],
        )
        # lhs operand layout: operand_index=0, k_width=8 (matches matmul.py
        # convention for BLOCK_K/BLOCK_N = 64 -> k_width = 64 // 8 = 8).
        lhs_layout = gl.DotOperandLayout(
            parent=acc_layout, operand_index=0, k_width=8
        )
        # rhs operand layout: operand_index=1.
        rhs_layout = gl.DotOperandLayout(
            parent=acc_layout, operand_index=1, k_width=8
        )
        # load_layout: 2-D BlockedLayout for Q/K/V/O tiles. 4 threads/col,
        # 8 elements per thread matches the fp16 16B coalescing target.
        load_layout = gl.BlockedLayout([1, 8], [4, 8], [_NUM_WARPS, 1], [1, 0])
        # row_layout: 1-D BlockedLayout for BLOCK_M-length row vectors
        # (m_i, l_i, alpha, lse, offs_m).
        row_layout = gl.BlockedLayout(
            [1], [32], [_NUM_WARPS], [0]
        )
        # col_layout: 1-D BlockedLayout for D-length column vectors (offs_d).
        col_layout = gl.BlockedLayout(
            [1], [32], [_NUM_WARPS], [0]
        )

        grid = (math.ceil(n / _BLOCK), b)
        _attn_fwd_kernel[grid](
            q3, k3, v3, o3, L, sm_scale,
            q3.stride(0), q3.stride(1), q3.stride(2),
            k3.stride(0), k3.stride(1), k3.stride(2),
            v3.stride(0), v3.stride(1), v3.stride(2),
            o3.stride(0), o3.stride(1), o3.stride(2),
            n,
            BLOCK_M=_BLOCK, BLOCK_N=_BLOCK, D=d, CAUSAL=causal,
            acc_layout=acc_layout,
            lhs_layout=lhs_layout,
            rhs_layout=rhs_layout,
            load_layout=load_layout,
            row_layout=row_layout,
            col_layout=col_layout,
            num_warps=_NUM_WARPS,
        )

        ctx.save_for_backward(q3, k3, v3, o3, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.shape = (z, h, n, d)
        return o3.reshape(z, h, n, d)

    @staticmethod
    def backward(ctx, do):  # noqa: ARG004
        raise NotImplementedError(
            "Gluon attention backward not yet implemented (P5)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attention(q, k, v, causal=False, sm_scale=None):
    """FlashAttention over 4-D (Z, H, N, D) tensor-core tensors (Gluon forward).

    Mirrors the Triton twin in triton_impl/attention.py. The backward pass
    is deferred (raises NotImplementedError until P5).

    Args:
        q: (Z, H, N, D) CUDA tensor, float16 or bfloat16.
        k: (Z, H, N, D) CUDA tensor, same dtype as q.
        v: (Z, H, N, D) CUDA tensor, same dtype as q.
        causal: If True, applies a causal (lower-triangular) mask.
        sm_scale: Softmax scale. Defaults to 1 / sqrt(D).

    Returns:
        (Z, H, N, D) output tensor, same dtype and device as q.
    """
    return _Attention.apply(q, k, v, causal, sm_scale)
