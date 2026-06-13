# Chapter 5: Matmul in Gluon — When Hand Control Loses to the Compiler

[Chapter 4](../04-matmul/) let Triton tile, group, and autotune the fp16
matmul. Here we hand-build the same matmul in Gluon — explicit MMA layouts and
`mma_v2` — and measure whether owning the pipeline beats the compiler.

It doesn't.

Sources:
[Triton twin](../../src/gluon_by_example/triton_impl/matmul.py) ·
[Gluon kernel](../../src/gluon_by_example/gluon_impl/matmul.py)

---

## The new machinery (vs chapter 3)

Chapter 3's softmax had reductions but no tensor-core MMA — `gl.reduce` and a
single `BlockedLayout` were the whole layout story. Matmul adds the part that
chapter said was still waiting: the MMA itself. Two new things show up here —
three coordinated MMA layouts and `mma_v2`.

The layouts are built host-side and passed in as constexprs:

```python
    acc_layout = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_NUM_WARPS, 1], instr_shape=[16, 8])
    lhs_layout = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=8)
    rhs_layout = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=8)
    load_layout = gl.BlockedLayout([1, 8], [4, 8], [_NUM_WARPS, 1], [1, 0])
```

`NVMMADistributedLayout` is the shape the tensor-core instruction leaves its
**fp32 accumulator** in — `instr_shape=[16, 8]` is the Ampere MMA tile. The two
`DotOperandLayout`s are the shapes the instruction wants its **A and B operands**
in: same parent (so they're coordinated with the accumulator), `operand_index`
0 vs 1 picking the left vs right operand, `k_width=8` packing eight fp16 along
the contraction dimension per lane. The `BlockedLayout` is the plain
coalesced-load layout — the shape a `gl.load` naturally produces. These are four
distinct views of the same tile data, and the kernel's job is to move between
them.

The K-loop is where that movement happens:

```python
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
```

Each tile lands in the `load_layout` (the coalesced shape `gl.load` produces),
and `convert_layout` reshapes it into the operand layout `mma_v2` expects — the
data is the same, the lane-to-element mapping changes. `mma_v2` then issues the
tensor-core multiply-accumulate, threading the running `acc` through every
iteration. Accumulation stays in fp32 across the whole K-loop; we downcast to
fp16 exactly once, on the way out — the same numerics as the Triton twin, which
also accumulates in fp32 and casts once at the store. The arithmetic is
identical to chapter 4. Only the scheduling differs.

---

## The narrow contract, stated honestly

The Gluon kernel takes fp16, tile-aligned inputs only — M and N divisible by
128, K by 64 — and does **no masking**. The wrapper guarantees in-bounds shapes,
so every load is unconditional. That is narrower than the Triton twin on
purpose: the Triton matmul masks its loads and stores, autotunes its tile shape,
and handles bf16 — it is the production gemm. This chapter is not trying to be
that. It is about the MMA/layout mechanics and the verdict, so the contract is
kept tight enough that the mechanics are the only thing on screen. Hand an
unaligned shape to `matmul` and it raises `ValueError` before launch — see
`test_gluon_rejects_unaligned` in `tests/test_matmul.py`.

---

## The verdict

The compiler wins this one, clearly. At N=8192 the hand-built Gluon kernel does
62.99 TFLOP/s against Triton's 113.12 — about **0.56×**. Across 2048 through 8192
the ratio sits between 0.54× and 0.57×; this is not a single bad point, it is the
shape of the whole curve.

The mechanism is scheduling, not arithmetic. Two things:

1. **Triton software-pipelines the loads.** Its pipeliner overlaps the *next*
   tile's `cp.async` loads with the *current* tile's MMA, staged several deep, so
   the tensor cores rarely wait on memory. Our floor kernel loads a tile, then
   computes on it, then loads the next — serially. The MMA units idle while the
   loads are in flight. That idle time is most of the gap.

2. **Consumer GPUs have no asynchronous MMA.** On this card the tensor-core
   instruction is synchronous — there is no async-MMA primitive to build an
   explicit producer/consumer ring against, the way you would on Hopper. So even
   hand-scheduling the loads can't fully decouple them from compute; there's no
   instruction to overlap the compute *against*.

Owning the layout taught us the mechanics — that was chapter 3's lesson, and it
held. Owning the *pipeline* here teaches a different lesson: this is where the
compiler is simply ahead, and saying so is the point of the chapter.

---

## What `cp.async` pipelining would add — and why it didn't help

The thing Triton's pipeliner does automatically is decouple loads from compute:
`cp.async` (and, on newer cards, TMA) lets a tile's load fire asynchronously and
land in shared memory while the tensor cores work on the previous tile. Build
enough stages and the memory latency hides entirely behind compute. That is
exactly the overlap our serial floor kernel is missing — so the obvious move is
to build those explicit `cp.async` pipelines by hand in Gluon and claw the gap
back.

We did — though not in this chapter's benchmark. In a longer experiment we built
those explicit `cp.async` and TMA pipelines by hand, and they did not beat this
simple floor: they matched or lost. Chapter 3 promised that chapter 5 "hands the
whole pipeline to Gluon," and this is the honest report from doing exactly that.
Hand-pipelining the loads doesn't pay on consumer silicon, because there is no
asynchronous MMA to overlap the loads *against* (the second mechanism above) —
and on this A6000 the TMA half of that machinery doesn't even exist yet; it
arrives with Hopper. The plain `mma_v2` floor is already the best a hand-built
Gluon matmul does here.

A forthcoming article carries the full story — the same question across more
GPUs and down the precision ladder (fp8, where the kernel turns feed-limited),
where the verdict gets more interesting.

---

## Benchmark

![matmul fp16: Triton vs Gluon vs cuBLAS](../../benchmarks/charts/matmul-gluon-nvidia-rtx-a6000.png)

Measured on the RTX A6000, fp16, square N×N, TFLOP/s:

| N | torch (cuBLAS) | triton | gluon |
|---|---|---|---|
| 256 | 5.06 | 5.64 | 2.43 |
| 512 | 31.02 | 30.99 | 11.52 |
| 1024 | 87.12 | 79.09 | 48.65 |
| 2048 | 105.26 | 102.70 | 55.47 |
| 4096 | 113.47 | 110.28 | 62.35 |
| 8192 | 110.51 | 113.12 | 62.99 |

At N=8192, gluon's 62.99 TFLOP/s is **0.56× of Triton's** 113.12 — the compiler
wins, and the gap is steady across the large sizes.

---

## Gotchas we hit

- **Layouts are compile-time `gl.constexpr`s.** All four are built host-side
  (where `_NUM_WARPS` is a plain `int`) and passed into the kernel as constexpr
  parameters — the same pattern chapter 3's softmax used for its single
  `BlockedLayout`.

- **`num_warps` is a reserved launch kwarg.** You cannot also pass the warp count
  in as a kernel constexpr — it collides with the launch argument. So the warp
  count is threaded through the *layouts* (their `warps_per_cta` fields) instead.
  The kernel wrapper says it directly: *"Layouts are built here (where num_warps
  is a plain int) and passed in as constexpr params... num_warps stays a reserved
  launch kwarg."*

- **Tile-aligned fp16 only.** Unaligned shapes or bf16 inputs raise `ValueError`
  before launch — `test_gluon_rejects_unaligned` and `test_gluon_rejects_bf16` in
  `tests/test_matmul.py` pin both. The Triton twin handles those cases; this
  kernel deliberately does not.

---

## Run it

```bash
pytest tests/test_matmul.py -q
python chapters/05-matmul-gluon/bench.py
```

Next: chapter 6 returns to Triton for flash attention — the kernel with real data reuse, where explicit control finally gets a fair fight.

*Written against Triton 3.7.0 (pip). Gluon is experimental; APIs move.*
