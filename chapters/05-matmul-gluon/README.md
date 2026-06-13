# Chapter 5: Matmul in Gluon — When Hand Control Loses to the Compiler

[Chapter 4](../04-matmul/) let Triton tile, group, and autotune the fp16
matmul. Here we hand-build the same matmul in Gluon — explicit MMA layouts and
`mma_v2` — and measure whether owning the pipeline beats the compiler.

It doesn't. And the *reason* it doesn't is not the one you'd guess — chasing it
down is the point of the chapter.

Sources:
[Triton twin](../../src/gluon_by_example/triton_impl/matmul.py) ·
[Gluon floor](../../src/gluon_by_example/gluon_impl/matmul.py) ·
[Gluon pipelined](../../src/gluon_by_example/gluon_impl/matmul_pipelined.py)

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
coalesced-load layout. These are distinct views of the same tile data, and the
kernel's job is to move between them.

The floor kernel's K-loop does exactly that, with no pipeline:

```python
    acc = gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, acc_layout)
    for k0 in range(0, K, BLOCK_K):
        a_tile = gl.load(a_ptr + offs_m[:, None] * K + offs_ka[None, :])
        b_tile = gl.load(b_ptr + offs_kb[:, None] * N + offs_n[None, :])
        acc = mma_v2(gl.convert_layout(a_tile, lhs_layout),
                     gl.convert_layout(b_tile, rhs_layout),
                     acc)
    out = gl.convert_layout(acc.to(gl.float16), load_layout)
```

`convert_layout` reshapes each loaded tile into the operand layout `mma_v2`
expects; accumulation stays in fp32 across the whole K-loop and downcasts to
fp16 once on the way out — the same numerics as the Triton twin. The arithmetic
is identical to chapter 4. Only the scheduling differs.

---

## The narrow contract, stated honestly

Both Gluon kernels here take fp16, tile-aligned inputs only — M and N divisible
by 128 (K by 64 for the floor, 32 for the pipelined version) — and do **no
masking**. The wrapper guarantees in-bounds shapes, so every load is
unconditional. That is narrower than the Triton twin on purpose: the Triton
matmul masks, autotunes its tile, and handles bf16 — it is the production gemm.
These chapters are about the MMA/layout/pipeline mechanics and the verdict, so
the contract is kept tight enough that the mechanics are the only thing on
screen. Hand an unaligned shape to either and it raises `ValueError` before
launch — see `test_gluon_rejects_unaligned` in `tests/test_matmul.py`.

---

## The verdict — and the surprise behind it

The floor loses to the compiler. Across the large sizes it runs about **0.55× of
Triton** (62.6 vs 113.2 TFLOP/s at N=8192). The obvious explanation — the one I
believed first — is that the floor *stalls*: it loads a tile, waits on global
memory, runs the MMA, repeats, with the tensor cores idle through every load.

**That explanation is wrong, and proving it wrong is the rest of this chapter.**

Here is the tell: the floor uses almost no shared memory (just scratch for
`convert_layout`), so it runs roughly **6 thread-blocks per SM**. With that many
blocks resident, Ampere's warp scheduler already hides the load latency — while
one block waits on a `gl.load`, it runs warps from another. The latency is
covered *for free, by occupancy*. So "idle MMA during loads" is not where the gap
comes from. We can prove it: build the pipeline that's supposed to fix the
stall, and watch it fail to close the gap.

## Act II: building the pipeline anyway

The textbook fix for load latency is a `cp.async` software pipeline — a
shared-memory ring buffer that prefetches the next K-tiles while the MMA works on
the current one. `cp.async` (asynchronous global→shared copy) exists on Ampere,
so this is buildable in Gluon:
[the pipelined kernel](../../src/gluon_by_example/gluon_impl/matmul_pipelined.py).

**First attempt: it backfired.** Built at the floor's `BLOCK_K=64` with a 3-stage
ring, the pipeline needs ~96 KB of shared memory — which collapses occupancy from
~6 blocks/SM to **1**. It ran at about **0.73× of the floor** — *slower* — even
though the overlap itself worked. The ring's shared memory evicted the very
co-resident blocks that were doing the latency hiding. (Reproduce it by setting
`_BLOCK_K = 64` in the pipelined kernel.) Occupancy, not pipeline logic, was the
binding resource.

**The fix: match the compiler's footprint.** Triton's autotuner doesn't pick
`BLOCK_K=64` here — its winning config is `BLOCK_K=32`, 4 stages, which buffers 3
K-tiles = 48 KB → 2 blocks/SM. Shrink `BLOCK_K` to 32 and the ring fits at 2
blocks/SM. Now the pipeline earns its keep over the floor:

| N | floor | pipelined | pipe / floor |
|---|---|---|---|
| 2048 | 55.4 | 74.4 | **1.34×** |
| 4096 | 62.3 | 79.4 | **1.28×** |
| 8192 | 62.6 | 56.5 | 0.90× |

At 2048 and 4096 the pipeline beats the floor by ~1.3× — real overlap, real win.
At 8192 it slips *below* the floor again (0.90×): the ring's lower occupancy
bites harder as the problem grows and there is more latency to hide.

**But it still loses to Triton** — 0.73× at 2048/4096, 0.50× at 8192 — even at
Triton's exact footprint.

## Where the last gap lives

With occupancy controlled (same `BLOCK_K`, same stage depth, same 48 KB /
2-blocks-per-SM as Triton) and bank conflicts handled (using `NVMMASharedLayout`
for the ring was worth ~50% on its own), the residual ~1.5× is *not* a
memory-budget problem. It is the parts of the compiler we did not reimplement:

- **L2-aware tile ordering.** The Triton kernel walks output tiles in
  `GROUP_M`-grouped super-rows so neighbors reuse A/B through L2 (chapter 4). The
  hand kernels use a plain 2-D grid with no grouping, so as the matrices outgrow
  L2 they re-fetch from DRAM far more. This is why the gap *widens* at N=8192.
- **Loop scheduling and register allocation.** Triton's backend interleaves the
  `cp.async` issue, the `wait`, and the MMA at instruction granularity and
  allocates registers around the pipeline; the hand loop issues them in source
  order.

That is the chapter's real lesson. The compiler's edge over hand-written Gluon
here is **not a primitive you can bolt on** — `cp.async` was right there, and we
used it. It is the *global co-optimization* of tile size, pipeline depth,
occupancy, tile-visit order, and instruction scheduling, searched automatically.
Hand control lets you match any one of those; matching all of them at once is
what the compiler does for free.

(The other way to overlap — the producer/consumer *warp-specialization* pattern —
isn't even available here: `warp_specialize` requires compute capability ≥ 9 and
the A6000 is 8.6, and it needs an asynchronous MMA to overlap against, which
consumer GPUs don't have. That is chapter 7's territory, on the 5090.)

---

## Benchmark

![matmul fp16: Triton vs Gluon (floor + cp.async) vs cuBLAS](../../benchmarks/charts/matmul-gluon-nvidia-rtx-a6000.png)

Measured on the RTX A6000, fp16, square N×N, TFLOP/s:

| N | torch (cuBLAS) | triton | gluon (floor) | gluon-pipe |
|---|---|---|---|---|
| 256 | 4.92 | 5.56 | 2.26 | 2.13 |
| 512 | 31.24 | 31.23 | 10.91 | 11.19 |
| 1024 | 87.17 | 70.02 | 47.42 | 47.17 |
| 2048 | 102.65 | 101.64 | 55.38 | 74.40 |
| 4096 | 112.71 | 110.26 | 62.25 | 79.42 |
| 8192 | 109.82 | 113.20 | 62.55 | 56.47 |

The climate is steady: `torch ≈ triton > pipelined > floor` at the mid sizes,
and the compiler holds a ~2× lead at 8192. Single mid-size points are weather —
`triton` at N=1024 reads 70.0 here and was 79.1 in chapter 5's earlier run;
neither the code nor the card changed. The reproducible signals are the ones the
chapter rests on: the floor at ~0.55× of Triton, the pipeline beating the floor
~1.3× at N=2048/4096, and the gap to Triton widening at 8192.

---

## Gotchas we hit

- **Footprint is the binding resource, not pipeline logic.** A deeper or
  bigger-tile `cp.async` ring can run *slower* than no pipeline, because the
  shared memory it needs evicts the co-resident blocks that were hiding latency
  through occupancy. The whole Act-II arc is this lesson.
- **`NVMMASharedLayout` swizzle width is capped by the inner contiguous dim.**
  At `BLOCK_K=32` the A tile's inner dim is 32 fp16 elements (64 B), so A takes a
  64 B swizzle while B (inner dim 128) keeps the wider 128 B one. Using the plain
  swizzled/no-swizzle layout instead cost ~50% to smem bank conflicts.
- **`cp.async` sync is `commit_group` / `wait_group`.** Each K-tile's copies are
  one committed group; `wait_group(STAGES-2)` keeps tile *k* ready while
  `STAGES-1` groups stay in flight. (mbarriers are the heavier alternative, for
  TMA on newer cards.)
- **Producer/consumer is not available here.** `warp_specialize` is gated at CC
  ≥ 9; the A6000 is 8.6. See chapter 7.
- **Tile-aligned fp16 only.** Both kernels raise `ValueError` on unaligned or
  bf16 inputs (`test_gluon_rejects_unaligned`, `test_gluon_rejects_bf16`). The
  Triton twin handles those; these deliberately do not.

---

## Run it

```bash
pytest tests/test_matmul.py -q
python chapters/05-matmul-gluon/bench.py
```

Next: chapter 6 returns to Triton for flash attention — the kernel with real data reuse, where explicit control finally gets a fair fight.

*Written against Triton 3.7.0 (pip). Gluon is experimental; APIs move.*
