# Three GPUs, one codebase: sm_86 vs sm_89 vs sm_120

The same four benchmarks — vector add, softmax (vs an unfused baseline),
softmax in Gluon, and an fp16 matmul — run unmodified on three compute
capabilities: an RTX A6000 (Ampere, 8.6), an RTX 4090 (Ada, 8.9), and an
RTX 5090 (Blackwell, 12.0). Same Triton 3.7.0 everywhere; nothing was
retuned per card. Comparison charts: `benchmarks/charts/*-compare.png`.

## The cards, as the driver reports them

Probed via `torch.cuda.get_device_properties` and Triton's driver utils —
not spec sheets:

| | RTX A6000 | RTX 4090 | RTX 5090 |
|---|---|---|---|
| Compute capability | 8.6 (GA102) | 8.9 (AD102) | 12.0 (GB202) |
| SMs | 84 | 128 | 170 |
| SM boost clock (probed) | 1.80 GHz | 2.63 GHz | 2.57 GHz |
| L2 cache | **6 MB** | **72 MB** | **96 MB** |
| Memory bus | 384-bit GDDR6 | 384-bit GDDR6X | 512-bit GDDR7 |
| Memory clock (probed) | 8.0 GHz | 10.5 GHz | 14.0 GHz |
| Theoretical bandwidth¹ | 768 GB/s | 1008 GB/s | 1792 GB/s |
| Max shared memory / CTA | 99 KB | 99 KB | 99 KB |
| Registers / SM | 64 K | 64 K | 64 K |

¹ Computed from probed clock × bus width × 2 (DDR): e.g. 8.001 GHz × 48 B
× 2 = 768.1 GB/s.

Three generations apart, the *programming model quantities* — shared
memory per block, register file, warp size — are identical. What changed
is everything the kernel doesn't see: SM count (2×), clocks (+43%),
memory technology (GDDR6 → 6X → 7), and L2 capacity (16×).

Note the compute-capability numbering: consumer Blackwell jumps from 8.9
to 12.0 — 9.0 (Hopper) and 10.0/10.3 (datacenter Blackwell) exist but
never shipped in a GeForce/workstation card.

## Vector add: the wall moves, the distance to it doesn't

Peak streaming bandwidth at 64 Mi elements (768 MB of traffic per call),
Triton kernel:

| | measured | theoretical | fraction |
|---|---|---|---|
| A6000 | 680.6 GB/s | 768.1 | **88.6%** |
| 4090 | 916.0 GB/s | 1008.1 | **90.9%** |
| 5090 | 1562.9 GB/s | 1792.1 | **87.2%** |

Three memory technologies, one law: a plain streaming kernel gets you
87–91% of paper bandwidth, no more. The 5090's GDDR7 delivers 2.3× the
A6000's GDDR6 — and the measured curves shift by exactly that factor.
torch, Triton, and Gluon are indistinguishable at the top end on all
three cards: there is nothing for a compiler to win when DRAM is the
bottleneck.

Two second-order effects are worth the look:

- **The latency floor does not improve.** At n = 4096 every card measures
  12–14 GB/s — the 5090 is actually the *slowest* of the three (11.7 vs
  the 4090's 12.4). A kernel launch costs what it costs; a new
  generation buys bandwidth, not latency.
- **Bigger cards need bigger inputs.** At 256 Ki elements the A6000 is
  already at 54% of its peak; the 4090 at 44%; the 5090 at 33%. The
  latency–bandwidth product grows every generation: the 5090 needs
  ~16 Mi elements to pass 97% of peak where the A6000 gets 95% from
  4 Mi. Saturating 170 SMs and a 512-bit bus takes parallelism that
  small tensors simply don't have.

## Softmax: where 72 MB of L2 shows up (and where it doesn't)

The benchmark is 4,096 rows × N columns, fp32. The fused kernels (torch,
Triton, Gluon) read the matrix twice and write it once regardless of N;
the `naive` provider is eight separate passes over global memory.

**Fused softmax rides the same wall as vector add**: peaks of ~665 /
~905 / ~1513 GB/s (87 / 90 / 84% of theoretical) — every line in the
fused subplot is just the vector-add story again.

**The naive provider is where the generations diverge.** The working set
of an unfused softmax is the input plus same-sized intermediates —
roughly 2× the tensor in flight between passes:

| N (tensor size) | A6000 (6 MB L2) | 4090 (72 MB) | 5090 (96 MB) |
|---|---|---|---|
| 1024 (16 MB) | 150.6 | 439.6 | 648.1 |
| 2048 (32 MB) | 157.8 | 411.5 | 728.7 |
| 4096 (64 MB) | 161.7 | **268.1** | **575.1** |
| 8192 (128 MB) | 163.2 | 230.1 | 402.0 |

The 4090 and 5090 hold 2.5–3× DRAM-only throughput while tensor +
intermediates fit in L2 (≲ 32 MB tensors), then fall off a cliff between
N = 2048 and N = 4096 — exactly where 2 × 64 MB stops fitting in 72/96 MB.
The A6000 has no cliff because it never had the cache: 6 MB holds
nothing, so it grinds at its flat ~150–165 GB/s for every size. Ada's
12× L2 jump (Ampere's 6 → 72 MB) is the single biggest *architectural*
change in this table, and it is visible only in the *bad* code.

That is the punchline worth keeping: **big L2 is a forgiveness feature.**
It rescues unfused, multi-pass code at sizes that happen to fit. The
fused kernel needs no forgiveness — its curve is smooth on all three
cards, and at N = 8192 it beats naive by 4.0× / 3.9× / 3.7×
(A6000 / 4090 / 5090). Fusion's value survives every hardware
generation; cache capacity only changes who gets hurt without it.

## Matmul: the only benchmark where silicon, not memory, is the story

fp16 square GEMMs, Triton kernel vs cuBLAS (`torch.matmul`), at
M = N = K = 8192 (arithmetic intensity ≈ 2,700 FLOP/byte — compute-bound
everywhere):

| | Triton | cuBLAS | Triton : A6000 ratio |
|---|---|---|---|
| A6000 | 110.7 TFLOP/s | 107.7 | 1.0× |
| 4090 | 173.9 | 167.0 | 1.57× |
| 5090 | 234.6 | 228.6 | 2.12× |

Two observations that the raw ratios hide:

- **Generational matmul scaling is slower than SM × clock scaling.**
  SM·GHz grows 1 : 2.22 : 2.88 across the three cards, but measured
  fp16 throughput grows 1 : 1.57 : 2.12. Per SM per clock (computed from
  the measurements): ~366 FMA-equivalents on the A6000 vs ~259 (4090)
  and ~269 (5090). The pro-class GA102 part runs a higher fp16
  tensor-core rate per SM than the GeForce parts that followed it — the
  consumer cards win on count and clock, not on per-SM punch.
  (Spec-sheet context: NVIDIA rates the A6000 at 154.8 fp16 Tensor
  TFLOPS, the 4090 at 165.2 with fp32 accumulate / 330 with fp16, the
  5090 at ~210 / 419. Measured numbers landing between the two
  accumulate ceilings are consistent with the reduced-precision
  reductions both cuBLAS and the autotuned Triton kernel are permitted
  to use; we report measurements and leave the accumulate-mode forensics
  to a chapter that needs them.)
- **Triton matches or beats cuBLAS on all three cards at large sizes**
  (within ±4% at 4096, ahead at 8192 on all three). The autotuned tile
  configurations carry across three architectures without per-card
  changes — the portability claim of the tile-level model, measured.
- The crossover behavior differs by card: at M = N = K = 1024 the 4090
  sits at 65–76 TFLOP/s while the 5090 already runs 120–128. Small GEMMs
  scale with clocks and scheduling, not peak tensor throughput.

## What a kernel author should take from 8.6 → 8.9 → 12.0

1. **Bandwidth-bound kernels port for free.** Same source, same tuning:
   87–91% of each card's theoretical bandwidth. Buy a faster card, get
   the multiplier on the sticker.
2. **The L2 cliff moved twice** (6 → 72 → 96 MB). If your workload's
   working set sits between 32 and 96 MB, *which* consumer card you run
   on changes unfused-code behavior by 2–3×. Fused code doesn't care.
3. **Latency didn't move.** Sub-microsecond kernels are the same speed
   on a 2020 card and a 2025 card; small-tensor workloads see almost
   none of the generational gain.
4. **Tensor-core generations don't scale like SM counts**, and pro vs
   consumer accumulate rates muddy spec-sheet comparisons. Measure; the
   paper TFLOPS rarely describe your kernel.
5. **Nothing in the programming model changed.** 99 KB shared memory,
   64 K registers, warp of 32 — a Triton/Gluon kernel written against
   the model is already written for all three cards. (What 12.0 *adds*
   — TMA, tensor memory, clusters — is opt-in capability, and the
   subject of later chapters.)

## Provenance

- Hardware tables: probed live on each card (`torch.cuda.get_device_properties`,
  `triton.runtime.driver.active.utils.get_device_properties`); theoretical
  bandwidth computed on-page from probed clocks. Spec-sheet TFLOPS are
  labeled as such and come from NVIDIA datasheets.
- Measurements: CSVs in `benchmarks/results/` (`triton.testing.do_bench`,
  which flushes L2 between timed iterations; the naive-softmax L2 reuse
  happens *within* one iteration, across its eight kernel launches).
- Machines: A6000 on the dev box (idle, 0 other compute PIDs); 4090 on a
  triple-4090 host, benchmarked on a GPU with no other processes while
  two sibling GPUs carried jobs (PCIe/host contention possible but the
  card itself was unshared); 5090 on a fully idle triple-5090 host —
  these are the committed numbers. A second 5090 run on a different,
  busier host agreed within 1–2% on every kernel, including matmul.
- Software: Triton 3.7.0 on all three cards (the Triton and Gluon
  columns are apples-to-apples). torch 2.12.0+cu130 on the A6000 and
  5090; the 4090's host driver (570.xx) caps it at torch 2.11.0+cu128,
  so its cuBLAS column is one library generation older.
- GPU selection guarded by `chapters/04-matmul/bench.py`'s idle-GPU
  check (scoped to the benchmark device). No `GBE_ALLOW_SHARED`
  override was used for any committed number.
