# src/gluon_by_example/triton_impl/attention.py
"""FlashAttention (forward + backward) in standard Triton.

Tiles over KV blocks with an online-softmax running rescale and two
tensor-core matmuls (QK^T, then P V), never materializing the N-by-N score
matrix. Forward saves the per-row logsumexp; the backward uses the FA2 split
(a preprocess kernel plus separate dk/dv and dq kernels, atomics-free).
Inputs are 4-D (Z, H, N, D); the wrapper views them as (Z*H, N, D).
"""

import math

import torch
import triton
import triton.language as tl

from gluon_by_example._validation import check_attention_inputs

_BLOCK = 64  # BLOCK_M == BLOCK_N keeps causal block alignment exact


@triton.jit
def _attn_fwd_kernel(Q, K, V, O, L, sm_scale,  # noqa: E741
                     stride_qb, stride_qm, stride_qd,
                     stride_kb, stride_kn, stride_kd,
                     stride_vb, stride_vn, stride_vd,
                     stride_ob, stride_om, stride_od,
                     N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     D: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_b = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_ptrs = Q + off_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + off_b * stride_kb + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)  # (D, BLOCK_N)
        qk = tl.dot(q, k) * sm_scale                              # (BLOCK_M, BLOCK_N) fp32
        qk = tl.where(offs_n[None, :] < N, qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + off_b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)  # (BLOCK_N, D)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    o_ptrs = O + off_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)
    tl.store(L + off_b * N + offs_m, lse, mask=offs_m < N)


@triton.jit
def _attn_bwd_preprocess(O, DO, Delta,  # noqa: E741
                         stride_ob, stride_om, stride_od,
                         N, BLOCK_M: tl.constexpr, D: tl.constexpr):
    start_m = tl.program_id(0)
    off_b = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    p = off_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o = tl.load(O + p, mask=offs_m[:, None] < N, other=0.0).to(tl.float32)  # noqa: E741
    do = tl.load(DO + p, mask=offs_m[:, None] < N, other=0.0).to(tl.float32)
    tl.store(Delta + off_b * N + offs_m, tl.sum(o * do, axis=1), mask=offs_m < N)


@triton.jit
def _attn_bwd_dkdv(Q, K, V, DO, DK, DV, L, Delta, sm_scale,  # noqa: E741
                   stride_b, stride_n, stride_d, N,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   D: tl.constexpr, CAUSAL: tl.constexpr):
    start_n = tl.program_id(0)
    off_b = tl.program_id(1)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    kv_mask = offs_n[:, None] < N
    k = tl.load(K + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=kv_mask, other=0.0)  # (BLOCK_N, D)
    v = tl.load(V + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=kv_mask, other=0.0)  # (BLOCK_N, D)
    dk = tl.zeros([BLOCK_N, D], tl.float32)
    dv = tl.zeros([BLOCK_N, D], tl.float32)
    lo = start_n * BLOCK_N if CAUSAL else 0
    for start_m in range(lo, N, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N
        q = tl.load(Q + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=m_mask[:, None], other=0.0)  # (BLOCK_M, D)
        do = tl.load(DO + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                     mask=m_mask[:, None], other=0.0)
        qkt = tl.dot(k, tl.trans(q)) * sm_scale            # (BLOCK_N, BLOCK_M) = S^T
        l_i = tl.load(L + off_b * N + offs_m, mask=m_mask, other=0.0)  # noqa: E741
        pT = tl.exp(qkt - l_i[None, :])                    # P^T
        valid = (offs_n[:, None] < N) & (m_mask[None, :])
        if CAUSAL:
            valid = valid & (offs_m[None, :] >= offs_n[:, None])
        pT = tl.where(valid, pT, 0.0)
        dv += tl.dot(pT.to(do.dtype), do)                  # (BLOCK_N, D)
        delta = tl.load(Delta + off_b * N + offs_m, mask=m_mask, other=0.0)
        dpT = tl.dot(v, tl.trans(do))                      # (BLOCK_N, BLOCK_M) = dP^T
        dsT = (pT * (dpT - delta[None, :]) * sm_scale)     # (BLOCK_N, BLOCK_M) = dS^T
        dk += tl.dot(dsT.to(q.dtype), q)                   # (BLOCK_N, D)
    tl.store(DK + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
             dk.to(DK.dtype.element_ty), mask=kv_mask)
    tl.store(DV + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
             dv.to(DV.dtype.element_ty), mask=kv_mask)


@triton.jit
def _attn_bwd_dq(Q, K, V, DO, DQ, L, Delta, sm_scale,  # noqa: E741
                 stride_b, stride_n, stride_d, N,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 D: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_b = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < N
    q = tl.load(Q + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                 mask=m_mask[:, None], other=0.0)
    l_i = tl.load(L + off_b * N + offs_m, mask=m_mask, other=0.0)  # noqa: E741
    delta = tl.load(Delta + off_b * N + offs_m, mask=m_mask, other=0.0)
    dq = tl.zeros([BLOCK_M, D], tl.float32)
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k = tl.load(K + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=n_mask[:, None], other=0.0)  # (BLOCK_N, D)
        v = tl.load(V + off_b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=n_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale         # (BLOCK_M, BLOCK_N)
        p = tl.exp(qk - l_i[:, None])
        valid = n_mask[None, :] & m_mask[:, None]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(do, tl.trans(v))                   # (BLOCK_M, BLOCK_N)
        ds = p * (dp - delta[:, None]) * sm_scale
        dq += tl.dot(ds.to(k.dtype), k)                # (BLOCK_M, D)
    tl.store(DQ + off_b * stride_b + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
             dq.to(DQ.dtype.element_ty), mask=m_mask[:, None])


def _shape3(t):
    z, h, n, d = t.shape
    return t.reshape(z * h, n, d)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        check_attention_inputs(q, k, v)
        z, h, n, d = q.shape
        sm_scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
        q3, k3, v3 = _shape3(q), _shape3(k), _shape3(v)
        o3 = torch.empty_like(q3)
        b = z * h
        L = torch.empty((b, n), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(n, _BLOCK), b)
        _attn_fwd_kernel[grid](
            q3, k3, v3, o3, L, sm_scale,
            q3.stride(0), q3.stride(1), q3.stride(2),
            k3.stride(0), k3.stride(1), k3.stride(2),
            v3.stride(0), v3.stride(1), v3.stride(2),
            o3.stride(0), o3.stride(1), o3.stride(2),
            n, BLOCK_M=_BLOCK, BLOCK_N=_BLOCK, D=d, CAUSAL=causal)
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
        grid_m = (triton.cdiv(n, _BLOCK), b)
        _attn_bwd_preprocess[grid_m](
            o3, do3, delta, o3.stride(0), o3.stride(1), o3.stride(2),
            n, BLOCK_M=_BLOCK, D=d)
        grid_n = (triton.cdiv(n, _BLOCK), b)
        _attn_bwd_dkdv[grid_n](
            q3, k3, v3, do3, dk, dv, L, delta, ctx.sm_scale,
            q3.stride(0), q3.stride(1), q3.stride(2), n,
            BLOCK_M=_BLOCK, BLOCK_N=_BLOCK, D=d, CAUSAL=ctx.causal)
        _attn_bwd_dq[grid_m](
            q3, k3, v3, do3, dq, L, delta, ctx.sm_scale,
            q3.stride(0), q3.stride(1), q3.stride(2), n,
            BLOCK_M=_BLOCK, BLOCK_N=_BLOCK, D=d, CAUSAL=ctx.causal)
        r = lambda t: t.reshape(z, h, n, d)  # noqa: E731
        return r(dq), r(dk), r(dv), None, None


def attention(q, k, v, causal=False, sm_scale=None):
    """FlashAttention over 4-D (Z, H, N, D) tensor-core tensors (differentiable)."""
    return _Attention.apply(q, k, v, causal, sm_scale)
