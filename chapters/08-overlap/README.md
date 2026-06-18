# Chapter 08: Overlap, keeping the GPU fed

Chapter 2 showed that a kernel launch returns immediately and the device runs behind the host.
This chapter cashes that in. The default rhythm of a GPU workload (copy inputs in, compute, copy
results out, repeat) leaves the copy engines idle while the SMs work and the SMs idle while the
copy engines work. Two host-side techniques close those gaps, and neither touches the kernel:
the compute is the book's Triton vector-add, unchanged.

Sources:
[overlap_demo.py](overlap_demo.py)

---

## Copy-compute overlap (streams + pinned memory)

A single serial pass pays for the H2D copy, then the compute, then the D2H copy, end to end. With
pinned host memory and several CUDA streams, the input copy of one chunk runs on the copy engine
while another chunk computes on the SMs and a third copies its result back. Throughput approaches
the slower of {copy, compute} instead of their sum.

```python
for i in range(n_chunks):
    with torch.cuda.stream(streams[i]):
        a = a_host[lo:hi].cuda(non_blocking=True)   # copy engine
        b = b_host[lo:hi].cuda(non_blocking=True)
        out_host[lo:hi].copy_(vector_add(a, b), non_blocking=True)  # SMs, then copy back
torch.cuda.synchronize()
```

The host tensors must be pinned (`pin_memory=True`); a pageable copy is synchronous and cannot
overlap. `overlapped_add` reproduces the serial result exactly.

---

## CUDA graphs (amortizing launch overhead)

For a workload that is many small kernels, the per-launch host cost (Python plus the driver)
dominates and the GPU idles between launches. Capturing the launch sequence once and replaying it
as a graph removes that overhead. The warmup pass matters: a graph cannot capture a kernel
compile, so the kernel must already be compiled before capture.

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = vector_add(a, b)
for _ in range(iters):
    graph.replay()
```

---

## The measured win (RTX A6000)

Copy-compute overlap, 16M fp32 elements (192MB round-tripped), effective end-to-end bandwidth:

| chunks | serial | overlap |
|---|---|---|
| 1  | 6.2 GB/s | 26.4 GB/s |
| 8  | 6.2 GB/s | 36.3 GB/s |
| 32 | 6.2 GB/s | 36.5 GB/s |

A single stream is transfer-bound at 6.2 GB/s; overlap reaches about 37 GB/s, a 5.9x effective
throughput gain, with the sweet spot around 16 to 32 streams.

CUDA graph replay vs eager launch, a 4K-element kernel launched back to back:

| launch count | eager | graph |
|---|---|---|
| 1000  | 7.97 us/launch | 2.97 us/launch |
| 10000 | 7.95 us/launch | 1.66 us/launch |

Eager launch holds steady near 8 us per launch; graph replay settles near 1.7 us, removing roughly
4.8x of the per-launch host overhead. The gap is the host cost the graph eliminates.

---

## Running it

```bash
python -m pytest tests/test_overlap.py     # correctness vs the serial baseline
python chapters/08-overlap/bench.py        # writes overlap-*.csv and charts (idle GPU only)
```

The benchmark refuses to run on a shared GPU; both overlap and launch-overhead numbers are
distorted by contention. Set `GBE_ALLOW_SHARED=1` to override with a warning.

---

## When it matters

Copy-compute overlap pays for memory-bound, chunked, transfer-heavy work, exactly the kernels left
of the roofline ridge. CUDA graphs pay for launch-bound workloads of many small kernels. Neither
helps a single large compute-bound kernel: there is nothing to hide behind it, and one launch has
no overhead to amortize.
