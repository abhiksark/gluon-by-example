# Chapter 10: Normalization in Gluon: gl.reduce and the Missing Atomic Floor

Chapter 9 built LayerNorm and RMSNorm in Triton and measured the atomic
weight-gradient floor against the two-stage-partial climb. This chapter
rewrites those kernels in Gluon. The forward and dx backward are
straightforward transliterations. The weight-gradient path is not: Gluon has
no `atomic_add`, so the atomic floor does not exist. Two-stage-partial is the
only option.

Sources:
[Gluon implementation](../../src/gluon_by_example/gluon_impl/normalization.py) ·
[Triton twin](../../src/gluon_by_example/triton_impl/normalization.py)

---

## What changes vs the Triton twin

**Forward.** The kernel structure is identical: one program per row, the whole
row in registers, mean/variance/rstd computed via reductions, output written
once. The syntax changes: `tl.sum` becomes `gl.reduce(..., combine_fn=_add_fn)`,
`tl.load`/`tl.store` become `gl.load`/`gl.store`, and layouts are passed in
explicitly as `gl.BlockedLayout` constexprs. The chapter is a close read of
what that substitution costs in verbosity versus what it gains in explicitness.

**dx backward.** The per-row input gradient is computed exactly as in the
Triton twin. The two row-level scalars (`c1`, `c2` for LayerNorm; `c1` only
for RMSNorm) come from `gl.reduce` instead of `tl.sum`. The surrounding
arithmetic is identical.

**dw / db (weight gradients).** Triton offered two paths: atomic floor and
two-stage partial. Gluon has no `atomic_add`, so the atomic floor is absent.
The two-stage grouped-partial is the only implementation here. Stage 1
accumulates strided rows into a `[GROUP_M, N]` partial buffer inside a while
loop; stage 2 reduces it to `[N]` in a separate kernel.

---

## The layout argument

Every Gluon kernel takes an explicit `layout: gl.constexpr` parameter built
on the host side. For normalization, the layout is a `gl.BlockedLayout` with
`threads_per_warp=[32]`, `warps_per_cta=[num_warps]`, and
`size_per_thread` set by dividing the block size across threads. The same
layout is threaded through the forward, dx backward, and weight-gradient
kernels so all tensor loads and stores see a consistent tile assignment.

---

## Unverified patterns

The Gluon normalization kernels carry four unverified call sites (flagged
inline in the source):

- `UNVERIFIED_SCALAR_STORE`: `gl.store` of a per-row scalar (mean, rstd) at a
  scalar pointer offset. Needs a GPU run to confirm correct behavior.
- `UNVERIFIED_SCALAR_LOAD`: `gl.load` of a scalar (mean, rstd) in the backward
  kernels, mirroring `tl.load(mean_ptr + row)` in the Triton twin.
- `UNVERIFIED_WHILE_LOOP`: dynamic while loop inside `@gluon.jit` for the
  strided row accumulation in stage 1.
- `UNVERIFIED_REDUCE_FOR_LOOP`: `for g in range(group_m)` inside `@gluon.jit`
  in the stage-2 reduce kernel, where `group_m` is a runtime value.

These patterns compile (static gate passes) but require a GPU run for
behavioral verification. The bench generates the evidence.

---

## Benchmark

Results are pending a GPU run. The GPU is currently unavailable; run the bench
to generate the chart:

```bash
python chapters/10-normalization-gluon/bench.py
```

The CSV will be written to `benchmarks/results/normalization-gluon-{gpu}.csv`
and the chart to `benchmarks/charts/normalization-gluon-{gpu}.png`.

Providers:
- `torch`: `F.layer_norm` forward only (the reference line).
- `gluon`: Gluon LayerNorm forward only.
- `gluon-dw-partial`: Gluon forward + backward, two-stage partial. Moves
  roughly 3x the bytes of the forward, so not directly comparable to the
  forward rows.

The chapter's claim to verify: Gluon forward matches (or approaches) the
Triton forward, and `gluon-dw-partial` is in the same range as
`triton-dw-partial` from chapter 9.

---

## Run it

```bash
pytest tests/test_normalization.py -v            # correctness (Triton + Gluon)
python chapters/10-normalization-gluon/bench.py  # regenerate CSV + chart
```

*Written against Triton 3.7.0 (pip). Gluon is experimental; APIs move.*
