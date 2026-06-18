# src/gluon_by_example/gluon_impl/attention.py
"""FlashAttention forward + backward in Gluon (experimental).

Tiles over KV blocks with an online-softmax running rescale (FA2 algorithm)
and two tensor-core matmuls (QK^T, then P V) via mma_v2. Never materializes
the N-by-N score matrix. Forward saves the per-row logsumexp L; the backward
uses the FA2 split: preprocess (Delta = rowsum(dO . O)), then dk/dv kernel
(loop over Q-blocks), then dq kernel (loop over KV-blocks). No atomics.

Inputs are 4-D (Z, H, N, D); the wrapper views them as (Z*H, N, D), exactly
matching the Triton twin in triton_impl/attention.py.

STATUS: static-checked only, not GPU-run. See module-level CONCERNS list
for gl.* calls that cannot be confirmed without a live Gluon runtime.

CONCERNS (flagged, not silently guessed):
  Forward concerns (from P4):
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

  Backward concerns (P5, higher risk):
  6. Transposed-operand mma_v2 in _attn_bwd_dkdv_kernel: the matmul S^T =
     K @ Q^T (BLOCK_N x BLOCK_M) is computed by loading Q with swapped strides
     (offs_d[:, None] * stride + offs_m[None, :] * stride_n) so it arrives as
     (D, BLOCK_M) rather than using a tl.trans equivalent. Whether Gluon's
     mma_v2 + DotOperandLayout pair accepts this transposed-stride pointer
     pattern in its rhs operand (operand_index=1) is unverified at static
     inspection time.
  7. Transposed-operand mma_v2 for dP^T = V @ dO^T (BLOCK_N x BLOCK_M):
     same transposed-stride pattern for dO in the rhs slot. Unverified.
  8. acc_layout_nt (NVMMADistributedLayout) for (BLOCK_N, BLOCK_M) tiles:
     the backward dk/dv kernels accumulate S^T, dP^T in (BLOCK_N, BLOCK_M)
     shape. Using an acc_layout with warps_per_cta=[_NUM_WARPS, 1] matches
     the forward convention, but the MMA tile is 16x8 and the outer dimensions
     are BLOCK_N x BLOCK_M (both 64). Whether the same NVMMADistributedLayout
     is valid for an (M=BLOCK_N, N=BLOCK_M) tile (i.e., whether
     warps_per_cta=[_NUM_WARPS, 1] is legal when the first dim is BLOCK_N=64
     rather than BLOCK_M=64) is unverified. It may need warps_per_cta=[1,
     _NUM_WARPS] or a different instr_shape.
  9. Per-block gl.store of dk/dv accumulator: the backward dkdv kernel
     accumulates dk, dv across all Q-blocks and stores them once per KV-block.
     The gl.store with out-of-bounds mask (offs_n[:, None] < N) on a
     (BLOCK_N, D) tile in acc_layout is assumed to behave like the Triton twin,
     but is unverified without a GPU run.
  10. Scalar gl.load of L and Delta per Q-block row: the pattern
      gl.load(L + off_b * N + offs_m, mask=offs_m < N, other=0.0) loads a
      BLOCK_M-length row_layout vector from a flat 1-D buffer. Whether Gluon's
      row_layout matches the Triton 1-D blocked load semantics is unverified.
  11. Delta broadcast delta[None, :] in dkdv / delta[:, None] in dq: the
      element-wise multiply pT * (dpT - delta[None, :]) mixes a (BLOCK_N,
      BLOCK_M) acc_layout_nt tile with a row_layout-derived broadcast. Whether
      Gluon propagates the broadcast correctly across a NVMMADistributedLayout
      tile is unverified (same class of risk as CONCERN 5 in the forward).
  12. Dynamic loop bound in dkdv: `lo = start_n * BLOCK_N if CAUSAL else 0`.
      The loop `for start_m in range(lo, N, BLOCK_M)` uses a runtime-computed
      start. Gluon's JIT range() handling for non-zero starts is assumed to
      mirror Triton's but is unverified for constexpr-only vs. dynamic starts.
  13. gl.store of dq accumulator inside dq kernel: after the inner KV-block
      loop, dq is stored to the DQ buffer in one shot per Q-block. Same class
      of risk as CONCERN 9 for the dkdv kernel.
  14. .to(gl.float16) cast before mma_v2 in backward: pT, p, dsT, ds are fp32
      NVMMADistributedLayout tiles cast to fp16 before feeding mma_v2. The
      backward matmuls (dsT @ Q, ds @ K) expect fp16 operands and fp32
      accumulation. Whether Gluon's .to(gl.float16) on an acc_layout_nt tile
      correctly re-materialises a DotOperandLayout-compatible representation
      before convert_layout is applied is unverified.
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
# Backward kernels
# ---------------------------------------------------------------------------

@gluon.jit
def _attn_bwd_preprocess_kernel(
        O, DO, Delta,  # noqa: E741
        stride_ob, stride_om, stride_od,
        N,
        BLOCK_M: gl.constexpr, D: gl.constexpr,
        load_layout: gl.constexpr,
        row_layout: gl.constexpr,
        col_layout: gl.constexpr):
    """Preprocess kernel: Delta[b, m] = rowsum(dO[b, m, :] * O[b, m, :]).

    One program owns BLOCK_M rows for one batch-head off_b. Mirrors
    _attn_bwd_preprocess in the Triton twin exactly.
    """
    start_m = gl.program_id(0)
    off_b = gl.program_id(1)
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    offs_d = gl.arange(0, D, layout=col_layout)
    p = off_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    # CONCERN 3 (mild): same pattern as forward; pointer arithmetic on
    # row_layout x col_layout aranges is assumed to produce a load_layout tile.
    o_tile = gl.load(O + p, mask=offs_m[:, None] < N, other=0.0).to(gl.float32)
    do_tile = gl.load(DO + p, mask=offs_m[:, None] < N, other=0.0).to(gl.float32)
    delta = gl.reduce(o_tile * do_tile, axis=1, combine_fn=_add_fn)
    # CONCERN 3: scalar 1-D gl.store to row_layout slice; mirrors Triton twin.
    gl.store(Delta + off_b * N + offs_m, delta, mask=offs_m < N)


@gluon.jit
def _attn_bwd_dkdv_kernel(
        Q, K, V, DO, DK, DV, L, Delta, sm_scale,  # noqa: E741
        stride_b, stride_n, stride_d, N,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, D: gl.constexpr,
        CAUSAL: gl.constexpr,
        acc_layout_nd: gl.constexpr,
        acc_layout_nt: gl.constexpr,
        lhs_layout_nd: gl.constexpr,
        rhs_layout_nd: gl.constexpr,
        lhs_layout_nt: gl.constexpr,
        rhs_layout_nt: gl.constexpr,
        load_layout: gl.constexpr,
        row_layout: gl.constexpr,
        col_layout: gl.constexpr):
    """dk/dv backward kernel: loop over Q-blocks for one KV-block.

    Mirrors _attn_bwd_dkdv in the Triton twin exactly:
      S^T  = K @ Q^T  (BLOCK_N, BLOCK_M)
      P^T  = exp(S^T - L[offs_m])
      dV  += P^T @ dO          (BLOCK_N, D)
      dP^T = V @ dO^T          (BLOCK_N, BLOCK_M)
      dS^T = P^T * (dP^T - Delta[offs_m]) * sm_scale
      dK  += dS^T @ Q          (BLOCK_N, D)

    CONCERN 8: acc_layout_nt is NVMMADistributedLayout for the (BLOCK_N, BLOCK_M)
    tiles (S^T, dP^T). Using warps_per_cta=[_NUM_WARPS, 1] for a BLOCK_N x BLOCK_M
    tile mirrors the forward but is unverified when the tile is not BLOCK_M x D.
    """
    start_n = gl.program_id(0)
    off_b = gl.program_id(1)
    offs_n = start_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=row_layout)
    offs_d = gl.arange(0, D, layout=col_layout)
    kv_mask = offs_n[:, None] < N

    # Load K, V tiles: both (BLOCK_N, D).
    k_tile = gl.load(
        K + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
        mask=kv_mask,
        other=0.0,
    )
    v_tile = gl.load(
        V + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
        mask=kv_mask,
        other=0.0,
    )

    # Accumulators (BLOCK_N, D) in fp32.
    dk = gl.full([BLOCK_N, D], 0.0, gl.float32, acc_layout_nd)
    dv = gl.full([BLOCK_N, D], 0.0, gl.float32, acc_layout_nd)

    lo = start_n * BLOCK_N if CAUSAL else 0
    # CONCERN 12: dynamic range start `lo` depends on runtime start_n when
    # CAUSAL=True. Gluon JIT range() is assumed to handle non-zero starts like
    # Triton, but this is unverified.
    for start_m in range(lo, N, BLOCK_M):
        offs_m = start_m + gl.arange(0, BLOCK_M, layout=row_layout)
        m_mask = offs_m < N

        # Load Q tile: (BLOCK_M, D).
        q_tile = gl.load(
            Q + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
            mask=m_mask[:, None],
            other=0.0,
        )
        # Load dO tile: (BLOCK_M, D).
        do_tile = gl.load(
            DO + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
            mask=m_mask[:, None],
            other=0.0,
        )

        # S^T = K @ Q^T: (BLOCK_N, BLOCK_M).
        # CONCERN 6: Q^T is loaded with transposed strides (offs_d[:, None],
        # offs_m[None, :]) so the rhs arrives as (D, BLOCK_M) at the mma_v2
        # rhs slot. This implements tl.trans(q) without a dedicated transpose
        # primitive. Whether Gluon's rhs DotOperandLayout (operand_index=1)
        # accepts a (D, BLOCK_M) shaped input for a (BLOCK_N, BLOCK_M) output
        # tile is unverified.
        qt_tile = gl.load(
            Q + off_b * stride_b + offs_d[:, None] * stride_d + offs_m[None, :] * stride_n,
            mask=m_mask[None, :],
            other=0.0,
        )
        qkt = mma_v2(
            gl.convert_layout(k_tile, lhs_layout_nt),
            gl.convert_layout(qt_tile, rhs_layout_nt),
            gl.full([BLOCK_N, BLOCK_M], 0.0, gl.float32, acc_layout_nt),
        )
        qkt = qkt * sm_scale  # S^T scaled

        # Load L[offs_m] (per-row logsumexp from forward).
        # CONCERN 10: scalar 1-D gl.load from row_layout slice.
        l_i = gl.load(L + off_b * N + offs_m, mask=m_mask, other=0.0)

        # P^T = exp(S^T - L[offs_m]): broadcast L across BLOCK_N rows.
        # CONCERN 11: pT * (dpT - delta[None, :]) mixes acc_layout_nt tile with
        # a row_layout-derived broadcast. Same cross-layout risk as CONCERN 5.
        pT = gl.exp(qkt - l_i[None, :])

        # Causal and padding mask.
        valid = (offs_n[:, None] < N) & (m_mask[None, :])
        if CAUSAL:
            valid = valid & (offs_m[None, :] >= offs_n[:, None])
        pT = gl.where(
            valid,
            pT,
            gl.full([BLOCK_N, BLOCK_M], 0.0, gl.float32, acc_layout_nt),
        )

        # dV += P^T @ dO: (BLOCK_N, D).
        # CONCERN 14: pT.to(gl.float16) on an acc_layout_nt tile before mma_v2.
        dv = mma_v2(
            gl.convert_layout(pT.to(gl.float16), lhs_layout_nt),
            gl.convert_layout(do_tile, rhs_layout_nd),
            dv,
        )

        # dP^T = V @ dO^T: (BLOCK_N, BLOCK_M).
        # CONCERN 7: dO^T loaded with transposed strides (offs_d[:, None],
        # offs_m[None, :]) so rhs is (D, BLOCK_M). Same risk as CONCERN 6.
        dot_tile = gl.load(
            DO + off_b * stride_b + offs_d[:, None] * stride_d + offs_m[None, :] * stride_n,
            mask=m_mask[None, :],
            other=0.0,
        )
        dpT = mma_v2(
            gl.convert_layout(v_tile, lhs_layout_nt),
            gl.convert_layout(dot_tile, rhs_layout_nt),
            gl.full([BLOCK_N, BLOCK_M], 0.0, gl.float32, acc_layout_nt),
        )

        # Load Delta[offs_m].
        # CONCERN 10: scalar 1-D gl.load from row_layout slice.
        delta = gl.load(Delta + off_b * N + offs_m, mask=m_mask, other=0.0)

        # dS^T = P^T * (dP^T - Delta[offs_m]) * sm_scale.
        # CONCERN 11: delta[None, :] broadcast across acc_layout_nt tile.
        dsT = pT * (dpT - delta[None, :]) * sm_scale

        # dK += dS^T @ Q: (BLOCK_N, D).
        # CONCERN 14: dsT.to(gl.float16) on acc_layout_nt before mma_v2.
        dk = mma_v2(
            gl.convert_layout(dsT.to(gl.float16), lhs_layout_nt),
            gl.convert_layout(q_tile, rhs_layout_nd),
            dk,
        )

    # Store dk, dv: (BLOCK_N, D).
    # CONCERN 9: gl.store of (BLOCK_N, D) acc_layout_nd tile with kv_mask.
    dk_out = gl.convert_layout(dk.to(gl.float16), load_layout)
    dv_out = gl.convert_layout(dv.to(gl.float16), load_layout)
    gl.store(
        DK + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
        dk_out,
        mask=kv_mask,
    )
    gl.store(
        DV + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
        dv_out,
        mask=kv_mask,
    )


@gluon.jit
def _attn_bwd_dq_kernel(
        Q, K, V, DO, DQ, L, Delta, sm_scale,  # noqa: E741
        stride_b, stride_n, stride_d, N,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, D: gl.constexpr,
        CAUSAL: gl.constexpr,
        acc_layout: gl.constexpr,
        lhs_layout: gl.constexpr,
        rhs_layout: gl.constexpr,
        load_layout: gl.constexpr,
        row_layout: gl.constexpr,
        col_layout: gl.constexpr):
    """dq backward kernel: loop over KV-blocks for one Q-block.

    Mirrors _attn_bwd_dq in the Triton twin exactly:
      S    = Q @ K^T  (BLOCK_M, BLOCK_N)
      P    = exp(S - L[offs_m])
      dP   = dO @ V^T (BLOCK_M, BLOCK_N)
      dS   = P * (dP - Delta[offs_m]) * sm_scale
      dQ  += dS @ K   (BLOCK_M, D)

    CONCERN 8 (dq): acc_layout is the same NVMMADistributedLayout as the
    forward (BLOCK_M, BLOCK_N) and (BLOCK_M, D) tiles, so no new shape
    question, but the transposed rhs slots for K^T and V^T (CONCERNS 6, 7
    class) apply here too.
    """
    start_m = gl.program_id(0)
    off_b = gl.program_id(1)
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    offs_d = gl.arange(0, D, layout=col_layout)
    m_mask = offs_m < N

    # Load Q, dO tiles: (BLOCK_M, D).
    q_tile = gl.load(
        Q + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
        mask=m_mask[:, None],
        other=0.0,
    )
    do_tile = gl.load(
        DO + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
        mask=m_mask[:, None],
        other=0.0,
    )

    # Load per-row L and Delta for this Q-block.
    # CONCERN 10: scalar 1-D gl.load from row_layout slice.
    l_i = gl.load(L + off_b * N + offs_m, mask=m_mask, other=0.0)
    delta = gl.load(Delta + off_b * N + offs_m, mask=m_mask, other=0.0)

    dq = gl.full([BLOCK_M, D], 0.0, gl.float32, acc_layout)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + gl.arange(0, BLOCK_N, layout=row_layout)
        n_mask = offs_n < N

        # Load K tile: (BLOCK_N, D). V is only needed in transposed form (vt_tile).
        k_tile = gl.load(
            K + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
            mask=n_mask[:, None],
            other=0.0,
        )

        # S = Q @ K^T: (BLOCK_M, BLOCK_N).
        # CONCERN 6 (dq variant): K^T loaded with transposed strides
        # (offs_d[:, None], offs_n[None, :]) arriving as (D, BLOCK_N) in rhs.
        kt_tile = gl.load(
            K + off_b * stride_b + offs_d[:, None] * stride_d + offs_n[None, :] * stride_n,
            mask=n_mask[None, :],
            other=0.0,
        )
        qk = mma_v2(
            gl.convert_layout(q_tile, lhs_layout),
            gl.convert_layout(kt_tile, rhs_layout),
            gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout),
        )
        qk = qk * sm_scale  # S scaled

        # P = exp(S - L): broadcast L across BLOCK_N columns.
        # CONCERN 11: l_i[:, None] broadcast across acc_layout tile.
        p = gl.exp(qk - l_i[:, None])

        # Padding + causal mask.
        valid = n_mask[None, :] & m_mask[:, None]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = gl.where(
            valid,
            p,
            gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout),
        )

        # dP = dO @ V^T: (BLOCK_M, BLOCK_N).
        # CONCERN 7 (dq variant): V^T loaded with transposed strides.
        vt_tile = gl.load(
            V + off_b * stride_b + offs_d[:, None] * stride_d + offs_n[None, :] * stride_n,
            mask=n_mask[None, :],
            other=0.0,
        )
        dp = mma_v2(
            gl.convert_layout(do_tile, lhs_layout),
            gl.convert_layout(vt_tile, rhs_layout),
            gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout),
        )

        # dS = P * (dP - Delta) * sm_scale.
        # CONCERN 11: delta[:, None] broadcast across acc_layout tile.
        ds = p * (dp - delta[:, None]) * sm_scale

        # dQ += dS @ K: (BLOCK_M, D).
        # CONCERN 14: ds.to(gl.float16) on acc_layout before mma_v2.
        dq = mma_v2(
            gl.convert_layout(ds.to(gl.float16), lhs_layout),
            gl.convert_layout(k_tile, rhs_layout),
            dq,
        )

    # Store dq: (BLOCK_M, D).
    # CONCERN 13: gl.store of (BLOCK_M, D) acc_layout tile with m_mask.
    dq_out = gl.convert_layout(dq.to(gl.float16), load_layout)
    gl.store(
        DQ + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
        dq_out,
        mask=m_mask[:, None],
    )


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
    """Autograd wrapper: forward + backward launch the Gluon FA kernels."""

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
    def backward(ctx, do):
        q3, k3, v3, o3, L = ctx.saved_tensors  # noqa: E741
        z, h, n, d = ctx.shape
        do3 = do.reshape(z * h, n, d).contiguous()
        b = z * h

        delta = torch.empty((b, n), device=q3.device, dtype=torch.float32)
        dq = torch.empty_like(q3)
        dk = torch.empty_like(k3)
        dv = torch.empty_like(v3)

        # Layout construction for backward kernels. acc_layout_nd accumulates
        # (BLOCK_N, D) or (BLOCK_M, D) tiles in fp32 via mma_v2.
        acc_layout_nd = gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[_NUM_WARPS, 1],
            instr_shape=[16, 8],
        )
        # CONCERN 8: acc_layout_nt is for (BLOCK_N, BLOCK_M) tiles. Using the
        # same warps_per_cta=[_NUM_WARPS, 1] as the forward is unverified for
        # a BLOCK_N x BLOCK_M (both 64) tile rather than BLOCK_M x D (64 x 64).
        acc_layout_nt = gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[_NUM_WARPS, 1],
            instr_shape=[16, 8],
        )
        # lhs/rhs for (BLOCK_N, D) output tiles (dK, dV accumulation).
        lhs_layout_nd = gl.DotOperandLayout(
            parent=acc_layout_nd, operand_index=0, k_width=8
        )
        rhs_layout_nd = gl.DotOperandLayout(
            parent=acc_layout_nd, operand_index=1, k_width=8
        )
        # lhs/rhs for (BLOCK_N, BLOCK_M) output tiles (S^T, dP^T accumulation).
        # CONCERN 8: k_width=8 mirrors matmul.py convention for D=64; whether
        # the same k_width applies when contracting over D for a (BLOCK_N,
        # BLOCK_M) output is unverified.
        lhs_layout_nt = gl.DotOperandLayout(
            parent=acc_layout_nt, operand_index=0, k_width=8
        )
        rhs_layout_nt = gl.DotOperandLayout(
            parent=acc_layout_nt, operand_index=1, k_width=8
        )
        # Forward layouts reused for the preprocess and dq kernels.
        acc_layout = gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[_NUM_WARPS, 1],
            instr_shape=[16, 8],
        )
        lhs_layout = gl.DotOperandLayout(
            parent=acc_layout, operand_index=0, k_width=8
        )
        rhs_layout = gl.DotOperandLayout(
            parent=acc_layout, operand_index=1, k_width=8
        )
        load_layout = gl.BlockedLayout([1, 8], [4, 8], [_NUM_WARPS, 1], [1, 0])
        row_layout = gl.BlockedLayout([1], [32], [_NUM_WARPS], [0])
        col_layout = gl.BlockedLayout([1], [32], [_NUM_WARPS], [0])

        grid_m = (math.ceil(n / _BLOCK), b)
        grid_n = (math.ceil(n / _BLOCK), b)

        # Step 1: preprocess -- Delta[b, m] = rowsum(dO[b, m, :] * O[b, m, :]).
        _attn_bwd_preprocess_kernel[grid_m](
            o3, do3, delta,
            o3.stride(0), o3.stride(1), o3.stride(2),
            n,
            BLOCK_M=_BLOCK, D=d,
            load_layout=load_layout,
            row_layout=row_layout,
            col_layout=col_layout,
            num_warps=_NUM_WARPS,
        )

        # Step 2: dk/dv -- one KV-block per program, loops over Q-blocks.
        _attn_bwd_dkdv_kernel[grid_n](
            q3, k3, v3, do3, dk, dv, L, delta, ctx.sm_scale,
            q3.stride(0), q3.stride(1), q3.stride(2), n,
            BLOCK_M=_BLOCK, BLOCK_N=_BLOCK, D=d, CAUSAL=ctx.causal,
            acc_layout_nd=acc_layout_nd,
            acc_layout_nt=acc_layout_nt,
            lhs_layout_nd=lhs_layout_nd,
            rhs_layout_nd=rhs_layout_nd,
            lhs_layout_nt=lhs_layout_nt,
            rhs_layout_nt=rhs_layout_nt,
            load_layout=load_layout,
            row_layout=row_layout,
            col_layout=col_layout,
            num_warps=_NUM_WARPS,
        )

        # Step 3: dq -- one Q-block per program, loops over KV-blocks.
        _attn_bwd_dq_kernel[grid_m](
            q3, k3, v3, do3, dq, L, delta, ctx.sm_scale,
            q3.stride(0), q3.stride(1), q3.stride(2), n,
            BLOCK_M=_BLOCK, BLOCK_N=_BLOCK, D=d, CAUSAL=ctx.causal,
            acc_layout=acc_layout,
            lhs_layout=lhs_layout,
            rhs_layout=rhs_layout,
            load_layout=load_layout,
            row_layout=row_layout,
            col_layout=col_layout,
            num_warps=_NUM_WARPS,
        )

        r = lambda t: t.reshape(z, h, n, d)  # noqa: E731
        return r(dq), r(dk), r(dv), None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attention(q, k, v, causal=False, sm_scale=None):
    """FlashAttention over 4-D (Z, H, N, D) tensor-core tensors (Gluon, differentiable).

    Mirrors the Triton twin in triton_impl/attention.py. Forward and backward
    both use Gluon kernels. STATUS: static-checked only, not GPU-run; see
    module-level CONCERNS for the full list of unverified gl.* calls.

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
