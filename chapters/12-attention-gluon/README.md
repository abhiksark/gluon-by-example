# Chapter 12: FlashAttention in Gluon

[Chapter 11](../11-attention/) built the same FlashAttention algorithm in standard Triton. This
chapter ports it to Gluon, using `mma_v2` and explicit layout constructors
(`NVMMADistributedLayout`, `DotOperandLayout`) in place of `tl.dot`. The goal is to see whether
explicit layout control pays off for a memory-bandwidth-bound, reduction-heavy kernel, and to
inventory the compiler interactions that matter.

Sources:
[Gluon implementation](../../src/gluon_by_example/gluon_impl/attention.py)

---

## The algorithm (same as chapter 11)

The forward pass tiles QK^T and PV using an online-softmax running rescale, loops over KV
blocks of width BLOCK_N=64, and never materializes the N-by-N score matrix. The backward
follows the FA2 three-kernel split: a preprocess kernel for Delta, a dk/dv kernel that loops
over Q-blocks per KV-block, and a dq kernel that loops over KV-blocks per Q-block. All three
kernels have the causal block-skip. The layout plumbing and kernel structure mirror the Triton
twin in chapter 11 exactly.

What changes in Gluon:

- `tl.dot(a, b)` becomes `mma_v2(gl.convert_layout(a, lhs_layout), gl.convert_layout(b, rhs_layout), acc)`.
- Running statistics (m_i, l_i, alpha, lse) use a 1-D `gl.BlockedLayout` (row_layout).
- `gl.reduce(x, axis=1, combine_fn=...)` replaces `tl.max` and `tl.sum`.
- Layout objects are constructed in Python and passed as `gl.constexpr` arguments to the kernel.

---

## GPU-unverified gl.* patterns (CONCERNS)

The Gluon forward and backward are **static-checked only, not GPU-run.** The module docstring
of `gluon_impl/attention.py` lists 15 specific CONCERNS; the most likely to require rework on
the first live run are:

- **CONCERN 1:** The second `mma_v2` (P @ V) reuses the same `lhs_layout` as QK^T. The
  contraction dimension is BLOCK_N=64 for PV vs. D=64 for QK^T. With `k_width=8` and both
  at 64 the values coincide, but the contract may differ in ways not visible at static time.
- **CONCERN 5 / 11:** `gl.where` conditions derived from `row_layout` arange vectors applied to
  `NVMMADistributedLayout` tiles. Whether Gluon broadcasts a 1-D condition across a 2-D
  distributed tile is unverified.
- **CONCERNS 6, 7:** Transposed-stride loads used to implement `tl.trans` for the backward
  S^T = K @ Q^T and dP^T = V @ dO^T. Whether Gluon's `DotOperandLayout` (operand_index=1)
  accepts a `(D, BLOCK_M)` shaped input for a `(BLOCK_N, BLOCK_M)` output tile is unverified.
- **CONCERN 8:** `acc_layout_nt` (NVMMADistributedLayout for BLOCK_N x BLOCK_M tiles in the
  backward dkdv kernel) uses `warps_per_cta=[_NUM_WARPS, 1]`. Whether this is valid when the
  outer dimension is BLOCK_N rather than BLOCK_M is unverified.
- **CONCERN 15:** DotOperandLayout parent-matching across the three backward kernels. Each
  `mma_v2` call requires the operand layouts to be parented to THAT call's accumulator. A
  static review pass found and corrected one mismatch; a live GPU run should re-verify all
  three backward kernels.

All 15 CONCERNS are documented inline at the relevant `gl.*` call sites. **Expect rework on the
first GPU run.**

---

## Expected codegen ceiling

The matmul chapter (chapter 5) measured Gluon at roughly 0.75x of Triton on Ampere, traced to
Gluon's JIT path skipping the TTGIR software-pipeliner pass. FlashAttention's QK^T and PV
matmuls are smaller tiles (BLOCK_M x BLOCK_N = 64 x 64 vs. 128 x 128 in matmul) and the kernel
carries significant reduction overhead (online-softmax rescale per KV block). Whether the
codegen ceiling is tighter or looser than 0.75x is an open question; results are pending a GPU
run.

---

## Run it

```bash
pytest tests/test_attention.py -q
python chapters/12-attention-gluon/bench.py
```

---

## Benchmark

Benchmark results are pending a GPU run. The bench sweeps sequence length N in [512, 1024, 2048,
4096, 8192] at fixed Z=2, H=8, D=64, comparing `torch` (`F.scaled_dot_product_attention`) and
`gluon` for both causal and non-causal variants. Metric is attention forward TFLOP/s charged at
the full non-causal FLOPs (4 * Z * H * N^2 * D multiply-adds). CSV goes to
`benchmarks/results/attention-gluon-{slug}.csv`.

To generate results and chart on your GPU:

```bash
python chapters/12-attention-gluon/bench.py
```

*Written against Triton 3.7.0 (pip). Gluon is experimental; APIs move.*
