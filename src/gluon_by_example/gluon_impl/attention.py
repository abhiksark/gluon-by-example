# src/gluon_by_example/gluon_impl/attention.py
"""FlashAttention forward + backward in Gluon (experimental).

Tiles over KV blocks with an online-softmax running rescale (FA2 algorithm)
and two tensor-core matmuls (QK^T, then P V) via mma_v2. Never materializes
the N-by-N score matrix. Forward saves the per-row logsumexp L; the backward
uses the FA2 split: preprocess (Delta = rowsum(dO . O)), then dk/dv kernel
(loop over Q-blocks), then dq kernel (loop over KV-blocks). No atomics.

Inputs are 4-D (Z, H, N, D); the wrapper views them as (Z*H, N, D), exactly
matching the Triton twin in triton_impl/attention.py.

STATUS: GPU-verified on an RTX A6000 (sm_86). Forward and backward, causal
and non-causal, match torch SDPA / autograd within fp16 tolerance.

The one layout rule that makes a Gluon FlashAttention express at all: every
index vector must be a gl.SliceLayout slice of the SAME 2-D parent it
indexes, never an independent 1-D BlockedLayout. For a (rows, cols) tile in
`parent`, the row index (used [:, None], axis 0) is SliceLayout(1, parent)
and the column index (used [None, :], axis 1) is SliceLayout(0, parent).
Two consequences worth internalising:
  - Loads/stores index load_layout (a BlockedLayout); the mma outputs, the
    masks, and the gl.where branches index acc_layout (NVMMADistributedLayout).
    The same logical offset (e.g. offs_m) therefore appears in several
    SliceLayout flavours -- one per parent-and-axis it is used against.
  - Per-row statistics inherit their layout from how they are produced or
    consumed. Reductions over axis 1 of an acc tile land in
    SliceLayout(1, acc_layout) and expand back via [:, None]; values that
    broadcast over the BLOCK_M axis of a transposed (BLOCK_N, BLOCK_M) tile
    live in SliceLayout(0, acc_layout_nt) and expand via [None, :].
Get those slices right and everything else follows: the transposed-operand
mma_v2 (S^T = K @ Q^T loaded with swapped strides), the (BLOCK_N, BLOCK_M)
acc_layout_nt tiles, the in-kernel fp16 casts, the scalar L/Delta loads, the
operand-parent matching (each convert_layout's DotOperandLayout is parented
to the accumulator of THAT mma_v2), and the dynamic causal loop start all
lower and compute correctly. The codegen ceiling from ch10 still applies:
this is the honest-best Gluon expression, not a speed record.
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
        load_layout: gl.constexpr):
    """Forward FA kernel: online-softmax tiling over KV blocks.

    One program (start_m, off_b) owns BLOCK_M query rows for one batch-head
    off_b. It iterates over KV blocks, maintaining running max m_i, running
    sum l_i, and accumulator acc in fp32. Two mma_v2 calls handle QK^T and
    PV respectively.

    Layout discipline follows matmul.py: every index vector is a
    gl.SliceLayout slice of the SAME 2-D parent it indexes, never an
    independent 1-D BlockedLayout (the CONCERN 16 fix). Load/store tiles use
    load_layout; the mma output and reductions use acc_layout, so per-row
    statistics (m_i, l_i, alpha, lse) live in SliceLayout(1, acc_layout)
    because they are reductions over axis 1 of an acc_layout tile.
    """
    start_m = gl.program_id(0)
    off_b = gl.program_id(1)

    # Index vectors as slices of their 2-D parents (matmul.py idiom).
    # For a (rows, cols) tile in `parent`: the row index (used [:, None],
    # axis 0) is SliceLayout(1, parent); the col index (used [None, :],
    # axis 1) is SliceLayout(0, parent).
    # Load-tile (load_layout) indices:
    offs_m_ld = start_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_layout))
    offs_d_c = gl.arange(0, D, gl.SliceLayout(0, load_layout))    # head-dim as a column
    offs_d_r = gl.arange(0, D, gl.SliceLayout(1, load_layout))    # head-dim as a row (K^T load)
    # Acc-tile (acc_layout) indices and the per-row stats layout.
    offs_m_ac = start_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, acc_layout))

    # Load Q tile: (BLOCK_M, D) -- loaded once, reused for all KV blocks.
    q_tile = gl.load(
        Q + off_b * stride_qb + offs_m_ld[:, None] * stride_qm + offs_d_c[None, :] * stride_qd,
        mask=offs_m_ld[:, None] < N,
        other=0.0,
    )

    # Running statistics (all fp32, in SliceLayout(1, acc_layout)).
    m_i = gl.full([BLOCK_M], -float("inf"), gl.float32, gl.SliceLayout(1, acc_layout))
    l_i = gl.full([BLOCK_M], 0.0, gl.float32, gl.SliceLayout(1, acc_layout))
    acc = gl.full([BLOCK_M, D], 0.0, gl.float32, acc_layout)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        # KV-block indices: as a load-tile row (V), as a load-tile col (K^T),
        # and as an acc-tile col (masks).
        offs_n_ld = start_n + gl.arange(0, BLOCK_N, gl.SliceLayout(1, load_layout))
        offs_n_c = start_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, load_layout))
        offs_n_ac = start_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, acc_layout))

        # Load K tile transposed: (D, BLOCK_N).
        k_tile = gl.load(
            K + off_b * stride_kb + offs_d_r[:, None] * stride_kd + offs_n_c[None, :] * stride_kn,
            mask=offs_n_c[None, :] < N,
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
        qk_masked = gl.where(
            offs_n_ac[None, :] < N,
            qk,
            gl.full([BLOCK_M, BLOCK_N], -float("inf"), gl.float32, acc_layout),
        )
        if CAUSAL:
            qk_masked = gl.where(
                offs_m_ac[:, None] >= offs_n_ac[None, :],
                qk_masked,
                gl.full([BLOCK_M, BLOCK_N], -float("inf"), gl.float32, acc_layout),
            )

        # Online-softmax update.
        m_ij = gl.maximum(m_i, gl.reduce(qk_masked, axis=1, combine_fn=_max_fn))
        p = gl.exp(qk_masked - m_ij[:, None])
        alpha = gl.exp(m_i - m_ij)

        # Rescale running sum and accumulator.
        l_i = l_i * alpha + gl.reduce(p, axis=1, combine_fn=_add_fn)
        acc = acc * alpha[:, None]

        # Load V tile: (BLOCK_N, D).
        v_tile = gl.load(
            V + off_b * stride_vb + offs_n_ld[:, None] * stride_vn + offs_d_c[None, :] * stride_vd,
            mask=offs_n_ld[:, None] < N,
            other=0.0,
        )

        # PV: acc += p @ V, accumulated in fp32.
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
    out = gl.convert_layout(acc.to(gl.float16), load_layout)
    gl.store(
        O + off_b * stride_ob + offs_m_ld[:, None] * stride_om + offs_d_c[None, :] * stride_od,
        out,
        mask=offs_m_ld[:, None] < N,
    )

    # Store per-row logsumexp L: shape (b, N), row-major. lse and offs_m_ac
    # share SliceLayout(1, acc_layout), so the 1-D store is layout-consistent.
    gl.store(L + off_b * N + offs_m_ac, lse, mask=offs_m_ac < N)


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
        load_layout: gl.constexpr):
    """Preprocess kernel: Delta[b, m] = rowsum(dO[b, m, :] * O[b, m, :]).

    One program owns BLOCK_M rows for one batch-head off_b. Mirrors
    _attn_bwd_preprocess in the Triton twin exactly. No mma here, so the row
    sum reduces over axis 1 of a load_layout tile: delta therefore lives in
    SliceLayout(1, load_layout), and the 1-D Delta store shares that layout.
    """
    start_m = gl.program_id(0)
    off_b = gl.program_id(1)
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_layout))
    offs_d = gl.arange(0, D, gl.SliceLayout(0, load_layout))
    p = off_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_tile = gl.load(O + p, mask=offs_m[:, None] < N, other=0.0).to(gl.float32)
    do_tile = gl.load(DO + p, mask=offs_m[:, None] < N, other=0.0).to(gl.float32)
    delta = gl.reduce(o_tile * do_tile, axis=1, combine_fn=_add_fn)
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
        load_layout: gl.constexpr):
    """dk/dv backward kernel: loop over Q-blocks for one KV-block.

    Mirrors _attn_bwd_dkdv in the Triton twin exactly:
      S^T  = K @ Q^T  (BLOCK_N, BLOCK_M)
      P^T  = exp(S^T - L[offs_m])
      dV  += P^T @ dO          (BLOCK_N, D)
      dP^T = V @ dO^T          (BLOCK_N, BLOCK_M)
      dS^T = P^T * (dP^T - Delta[offs_m]) * sm_scale
      dK  += dS^T @ Q          (BLOCK_N, D)

    The (BLOCK_N, BLOCK_M) score-transpose tiles use acc_layout_nt; the
    (BLOCK_N, D) gradient accumulators use acc_layout_nd. Every index vector
    is a SliceLayout slice of the parent it indexes (load_layout for memory
    tiles, acc_layout_nt for the score tiles). Per-row L/Delta broadcast over
    the BLOCK_M axis of an acc_nt tile, so they live in
    SliceLayout(0, acc_layout_nt) -- expanded via [None, :].
    """
    start_n = gl.program_id(0)
    off_b = gl.program_id(1)

    # KV (BLOCK_N) indices: as a load-tile row, and as an acc_nt row.
    offs_n_ld = start_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(1, load_layout))
    offs_n_nt = start_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(1, acc_layout_nt))
    # Head-dim indices: as a load-tile column, and as a transposed-load row.
    offs_d_c = gl.arange(0, D, gl.SliceLayout(0, load_layout))
    offs_d_r = gl.arange(0, D, gl.SliceLayout(1, load_layout))
    kv_mask = offs_n_ld[:, None] < N

    # Load K, V tiles: both (BLOCK_N, D).
    k_tile = gl.load(
        K + off_b * stride_b + offs_n_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
        mask=kv_mask,
        other=0.0,
    )
    v_tile = gl.load(
        V + off_b * stride_b + offs_n_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
        mask=kv_mask,
        other=0.0,
    )

    # Accumulators (BLOCK_N, D) in fp32.
    dk = gl.full([BLOCK_N, D], 0.0, gl.float32, acc_layout_nd)
    dv = gl.full([BLOCK_N, D], 0.0, gl.float32, acc_layout_nd)

    lo = start_n * BLOCK_N if CAUSAL else 0
    for start_m in range(lo, N, BLOCK_M):
        # Q (BLOCK_M) indices: as a load-tile row, as a transposed-load column,
        # and as an acc_nt column (the broadcast axis for L/Delta and masks).
        offs_m_ld = start_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_layout))
        offs_m_c = start_m + gl.arange(0, BLOCK_M, gl.SliceLayout(0, load_layout))
        offs_m_nt = start_m + gl.arange(0, BLOCK_M, gl.SliceLayout(0, acc_layout_nt))

        # Load Q tile: (BLOCK_M, D).
        # NOTE (bandwidth opt): Q is loaded twice per iteration -- once here as
        # q_tile (row-major) and once below as qt_tile (transposed strides). A
        # single load + in-register transpose would halve Q bandwidth.
        q_tile = gl.load(
            Q + off_b * stride_b + offs_m_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
            mask=offs_m_ld[:, None] < N,
            other=0.0,
        )
        # Load dO tile: (BLOCK_M, D).
        do_tile = gl.load(
            DO + off_b * stride_b + offs_m_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
            mask=offs_m_ld[:, None] < N,
            other=0.0,
        )

        # S^T = K @ Q^T: (BLOCK_N, BLOCK_M). Q^T loaded transposed as (D, BLOCK_M).
        qt_tile = gl.load(
            Q + off_b * stride_b + offs_d_r[:, None] * stride_d + offs_m_c[None, :] * stride_n,
            mask=offs_m_c[None, :] < N,
            other=0.0,
        )
        qkt = mma_v2(
            gl.convert_layout(k_tile, lhs_layout_nt),
            gl.convert_layout(qt_tile, rhs_layout_nt),
            gl.full([BLOCK_N, BLOCK_M], 0.0, gl.float32, acc_layout_nt),
        )
        qkt = qkt * sm_scale  # S^T scaled

        # Load L[offs_m] (per-row logsumexp from forward).
        l_i = gl.load(L + off_b * N + offs_m_nt, mask=offs_m_nt < N, other=0.0)

        # P^T = exp(S^T - L[offs_m]): broadcast L across BLOCK_N rows.
        pT = gl.exp(qkt - l_i[None, :])

        # Causal and padding mask.
        valid = (offs_n_nt[:, None] < N) & (offs_m_nt[None, :] < N)
        if CAUSAL:
            valid = valid & (offs_m_nt[None, :] >= offs_n_nt[:, None])
        pT = gl.where(
            valid,
            pT,
            gl.full([BLOCK_N, BLOCK_M], 0.0, gl.float32, acc_layout_nt),
        )

        # dV += P^T @ dO: (BLOCK_N, D). Operands parented to dv's acc_layout_nd.
        dv = mma_v2(
            gl.convert_layout(pT.to(gl.float16), lhs_layout_nd),
            gl.convert_layout(do_tile, rhs_layout_nd),
            dv,
        )

        # dP^T = V @ dO^T: (BLOCK_N, BLOCK_M). dO^T loaded transposed as (D, BLOCK_M).
        dot_tile = gl.load(
            DO + off_b * stride_b + offs_d_r[:, None] * stride_d + offs_m_c[None, :] * stride_n,
            mask=offs_m_c[None, :] < N,
            other=0.0,
        )
        dpT = mma_v2(
            gl.convert_layout(v_tile, lhs_layout_nt),
            gl.convert_layout(dot_tile, rhs_layout_nt),
            gl.full([BLOCK_N, BLOCK_M], 0.0, gl.float32, acc_layout_nt),
        )

        # Load Delta[offs_m].
        delta = gl.load(Delta + off_b * N + offs_m_nt, mask=offs_m_nt < N, other=0.0)

        # dS^T = P^T * (dP^T - Delta[offs_m]) * sm_scale.
        dsT = pT * (dpT - delta[None, :]) * sm_scale

        # dK += dS^T @ Q: (BLOCK_N, D). Operands parented to dk's acc_layout_nd.
        dk = mma_v2(
            gl.convert_layout(dsT.to(gl.float16), lhs_layout_nd),
            gl.convert_layout(q_tile, rhs_layout_nd),
            dk,
        )

    # Store dk, dv: (BLOCK_N, D).
    dk_out = gl.convert_layout(dk.to(gl.float16), load_layout)
    dv_out = gl.convert_layout(dv.to(gl.float16), load_layout)
    gl.store(
        DK + off_b * stride_b + offs_n_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
        dk_out,
        mask=kv_mask,
    )
    gl.store(
        DV + off_b * stride_b + offs_n_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
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
        load_layout: gl.constexpr):
    """dq backward kernel: loop over KV-blocks for one Q-block.

    Mirrors _attn_bwd_dq in the Triton twin exactly:
      S    = Q @ K^T  (BLOCK_M, BLOCK_N)
      P    = exp(S - L[offs_m])
      dP   = dO @ V^T (BLOCK_M, BLOCK_N)
      dS   = P * (dP - Delta[offs_m]) * sm_scale
      dQ  += dS @ K   (BLOCK_M, D)

    Same SliceLayout discipline as the forward: the (BLOCK_M, BLOCK_N) and
    (BLOCK_M, D) tiles use acc_layout; per-row L/Delta broadcast over the
    BLOCK_N axis, so they live in SliceLayout(1, acc_layout) -- expanded via
    [:, None]. K^T and V^T are loaded with transposed strides as (D, BLOCK_N).
    """
    start_m = gl.program_id(0)
    off_b = gl.program_id(1)

    # Q (BLOCK_M) indices: as a load-tile row, and as an acc row.
    offs_m_ld = start_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_layout))
    offs_m_ac = start_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, acc_layout))
    # Head-dim indices: as a load-tile column, and as a transposed-load row.
    offs_d_c = gl.arange(0, D, gl.SliceLayout(0, load_layout))
    offs_d_r = gl.arange(0, D, gl.SliceLayout(1, load_layout))

    # Load Q, dO tiles: (BLOCK_M, D).
    q_tile = gl.load(
        Q + off_b * stride_b + offs_m_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
        mask=offs_m_ld[:, None] < N,
        other=0.0,
    )
    do_tile = gl.load(
        DO + off_b * stride_b + offs_m_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
        mask=offs_m_ld[:, None] < N,
        other=0.0,
    )

    # Load per-row L and Delta for this Q-block (broadcast over BLOCK_N columns).
    l_i = gl.load(L + off_b * N + offs_m_ac, mask=offs_m_ac < N, other=0.0)
    delta = gl.load(Delta + off_b * N + offs_m_ac, mask=offs_m_ac < N, other=0.0)

    dq = gl.full([BLOCK_M, D], 0.0, gl.float32, acc_layout)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        # KV (BLOCK_N) indices: as a load-tile row, a transposed-load column,
        # and an acc column.
        offs_n_ld = start_n + gl.arange(0, BLOCK_N, gl.SliceLayout(1, load_layout))
        offs_n_c = start_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, load_layout))
        offs_n_ac = start_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, acc_layout))

        # Load K tile: (BLOCK_N, D). V is only needed in transposed form (vt_tile).
        k_tile = gl.load(
            K + off_b * stride_b + offs_n_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
            mask=offs_n_ld[:, None] < N,
            other=0.0,
        )

        # S = Q @ K^T: (BLOCK_M, BLOCK_N). K^T loaded transposed as (D, BLOCK_N).
        kt_tile = gl.load(
            K + off_b * stride_b + offs_d_r[:, None] * stride_d + offs_n_c[None, :] * stride_n,
            mask=offs_n_c[None, :] < N,
            other=0.0,
        )
        qk = mma_v2(
            gl.convert_layout(q_tile, lhs_layout),
            gl.convert_layout(kt_tile, rhs_layout),
            gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout),
        )
        qk = qk * sm_scale  # S scaled

        # P = exp(S - L): broadcast L across BLOCK_N columns.
        p = gl.exp(qk - l_i[:, None])

        # Padding + causal mask.
        valid = (offs_n_ac[None, :] < N) & (offs_m_ac[:, None] < N)
        if CAUSAL:
            valid = valid & (offs_m_ac[:, None] >= offs_n_ac[None, :])
        p = gl.where(
            valid,
            p,
            gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout),
        )

        # dP = dO @ V^T: (BLOCK_M, BLOCK_N). V^T loaded transposed as (D, BLOCK_N).
        vt_tile = gl.load(
            V + off_b * stride_b + offs_d_r[:, None] * stride_d + offs_n_c[None, :] * stride_n,
            mask=offs_n_c[None, :] < N,
            other=0.0,
        )
        dp = mma_v2(
            gl.convert_layout(do_tile, lhs_layout),
            gl.convert_layout(vt_tile, rhs_layout),
            gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout),
        )

        # dS = P * (dP - Delta) * sm_scale.
        ds = p * (dp - delta[:, None]) * sm_scale

        # dQ += dS @ K: (BLOCK_M, D).
        dq = mma_v2(
            gl.convert_layout(ds.to(gl.float16), lhs_layout),
            gl.convert_layout(k_tile, rhs_layout),
            dq,
        )

    # Store dq: (BLOCK_M, D).
    dq_out = gl.convert_layout(dq.to(gl.float16), load_layout)
    gl.store(
        DQ + off_b * stride_b + offs_m_ld[:, None] * stride_n + offs_d_c[None, :] * stride_d,
        dq_out,
        mask=offs_m_ld[:, None] < N,
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
        # 8 elements per thread matches the fp16 16B coalescing target. Every
        # 1-D index is derived inside the kernel as a SliceLayout of this or
        # acc_layout, so no standalone row/col layouts are passed.
        load_layout = gl.BlockedLayout([1, 8], [4, 8], [_NUM_WARPS, 1], [1, 0])

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

        grid_m = (math.ceil(n / _BLOCK), b)
        grid_n = (math.ceil(n / _BLOCK), b)

        # Step 1: preprocess -- Delta[b, m] = rowsum(dO[b, m, :] * O[b, m, :]).
        _attn_bwd_preprocess_kernel[grid_m](
            o3, do3, delta,
            o3.stride(0), o3.stride(1), o3.stride(2),
            n,
            BLOCK_M=_BLOCK, D=d,
            load_layout=load_layout,
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
    both use Gluon kernels. GPU-verified against torch SDPA / autograd; see
    the module docstring for the SliceLayout discipline that makes it express.

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
