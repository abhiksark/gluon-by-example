# Chapter 9: Normalization in Triton: The Weight-Gradient Reduction

Chapters 2 and 4 covered two bandwidth-bound kernels: softmax and matmul.
Normalization is the third. The forward pass is a fused-row kernel, essentially
the same shape as softmax. The backward pass adds a wrinkle that softmax does
not have: **the weight gradient is a cross-row reduction**, and how you
implement that reduction is the chapter's central question.

Sources:
[Triton implementation](../../src/gluon_by_example/triton_impl/normalization.py)

---

## What LayerNorm and RMSNorm compute

Given an input matrix `x` of shape `[M, N]`, weights `w` of shape `[N]`, and
biases `b` of shape `[N]` (LayerNorm only):

**LayerNorm** normalizes each row to zero mean and unit variance, then applies
a learned affine transform:

```
mean  = sum(x[row]) / N
var   = sum((x[row] - mean)^2) / N
rstd  = 1 / sqrt(var + eps)
y     = (x - mean) * rstd * w + b
```

**RMSNorm** skips the mean subtraction, normalizing by root-mean-square instead:

```
ms    = sum(x[row]^2) / N
rstd  = 1 / sqrt(ms + eps)
y     = x * rstd * w
```

RMSNorm is cheaper by one sum and one subtraction per row. Both appear in
modern language model architectures (LN is the original Transformer norm; RMS
is used in LLaMA, Gemma, and others).

---

## The fused-row forward

The forward pass reuses the same strategy as softmax: one program per row,
the whole row loaded into registers at once. Because `N` can vary, the block
size is rounded up to the next power of two and the excess lanes are masked
(`other=0.0`). Statistics (mean, rstd) are computed entirely in registers and
written once to per-row scratch buffers; the output `y` is written once.
Total traffic: read `x`, `w`, `b`; write `y`, `mean`, `rstd`. The mean and
rstd writes are `M` scalars, negligible next to the `M x N` matrix.

---

## The backward pass

The backward has two parts with very different traffic patterns.

**dx (per-row input gradient).** Given upstream gradient `dy`, the chain rule
through the normalization statistics produces two row-level scalars `c1` and
`c2` (for LayerNorm; RMSNorm needs only `c1`), which are then used in a
simple elementwise formula:

```
x_hat = (x - mean) * rstd
wdy   = w * dy
c1    = sum(x_hat * wdy) / N
c2    = sum(wdy) / N
dx    = (wdy - x_hat * c1 - c2) * rstd
```

This is per-row: each program reads `dy`, `x`, `w`, `mean`, `rstd` for its
row and writes `dx`. Same fused-row shape as the forward.

**dw and db (weight gradients).** The weight `w` is shared across all `M`
rows. Each row contributes `dy[row] * x_hat[row]` to `dw` (and `dy[row]` to
`db`). Computing `dw` requires summing these contributions from every row: a
**cross-row reduction**. This is the hard part.

---

## The atomic floor vs two-stage partial

The direct implementation is to have each program atomically add its row's
contribution into a shared `dw` accumulator:

```python
tl.atomic_add(dw_ptr + cols, dy * x_hat, mask=mask)
```

This is correct but contention-bound: when many programs try to atomic-add to
the same `N` columns simultaneously, the hardware serializes them. At large
`N` and large `M`, this turns into a traffic bottleneck. This is the
**atomic floor**.

The two-stage partial approach avoids the contention. Instead of one shared
accumulator:

1. **Stage 1 (partial).** `GROUP_M` programs each own one partial row and
   accumulate every `GROUP_M`-th input row into it with a strided while loop.
   No two programs write to the same location, so no atomics are needed.
   Output: a `[GROUP_M, N]` partial buffer.

2. **Stage 2 (reduce).** A second small kernel sums the `GROUP_M` partial rows
   down to `[N]`, one column tile per program.

The tradeoff: stage 1 launches far fewer programs (`GROUP_M = 64` instead of
`M = 4096`), so each program processes more rows in sequence. The extra reads
are cheaper than the atomic contention they replace at large `M * N`.

The bench measures both. Results are pending a GPU run (see below).

---

## Benchmark

Results are pending a GPU run. The GPU is currently unavailable; run the bench
to generate the chart:

```bash
python chapters/09-normalization/bench.py
```

The CSV will be written to `benchmarks/results/normalization-{gpu}.csv` and
the chart to `benchmarks/charts/normalization-{gpu}.png`.

Note on the backward providers: `triton-dw-atomic` and `triton-dw-partial`
time a full forward + backward pass. The backward moves roughly 3x the bytes
of the forward (reading x, dy, mean, rstd; writing dx, dw, db), so their
effective-bandwidth numbers are not directly comparable to the forward rows.
The comparison between the two backward providers isolates the cost of
atomic-add versus two-stage-partial for the weight-gradient reduction.

---

## Run it

```bash
pytest tests/test_normalization.py -v       # correctness
python chapters/09-normalization/bench.py   # regenerate CSV + chart
```

Next: [chapter 10](../10-normalization-gluon/) rewrites these kernels in
Gluon, where `gl.reduce` replaces `tl.sum/tl.max` and the atomic floor is
gone: Gluon's weight-gradient path is two-stage-partial only.

*Written against Triton 3.7.0 (pip). Gluon is experimental; APIs move.*
