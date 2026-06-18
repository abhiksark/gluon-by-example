# chapters/08-overlap/overlap_demo.py
"""Host-side overlap demos reusing the vector-add kernel.

Two ways to keep the GPU fed: (1) copy-compute overlap with pinned host memory
and multiple CUDA streams, and (2) a CUDA graph that replays a captured launch
sequence to remove per-launch host overhead. The compute kernel is the book's
Triton vector-add, unchanged; everything here is host orchestration.
"""

import torch

from gluon_by_example.triton_impl.vector_add import vector_add


def serial_add(a_host: torch.Tensor, b_host: torch.Tensor) -> torch.Tensor:
    """Baseline: one H2D copy, one compute, one D2H copy, no overlap."""
    out = vector_add(a_host.cuda(), b_host.cuda())
    return out.cpu()


def overlapped_add(a_host: torch.Tensor, b_host: torch.Tensor,
                   n_chunks: int = 8) -> torch.Tensor:
    """Copy-compute overlap across streams; inputs must be pinned CPU tensors.

    Each chunk's H2D copy, vector-add, and D2H copy run on their own stream, so
    with pinned memory the copy engines stay busy while other chunks compute.

    Args:
        a_host: Pinned 1-D CPU tensor.
        b_host: Pinned 1-D CPU tensor, same shape and dtype as a_host.
        n_chunks: Number of chunks/streams to split the work across.

    Returns:
        Pinned CPU tensor with the elementwise sum.
    """
    n = a_host.numel()
    chunk = (n + n_chunks - 1) // n_chunks
    out_host = torch.empty(n, dtype=a_host.dtype, pin_memory=True)
    streams = [torch.cuda.Stream() for _ in range(n_chunks)]
    for i in range(n_chunks):
        lo, hi = i * chunk, min((i + 1) * chunk, n)
        if lo >= hi:
            break
        with torch.cuda.stream(streams[i]):
            a = a_host[lo:hi].cuda(non_blocking=True)
            b = b_host[lo:hi].cuda(non_blocking=True)
            out_host[lo:hi].copy_(vector_add(a, b), non_blocking=True)
    torch.cuda.synchronize()
    return out_host


def graphed_repeat(a: torch.Tensor, b: torch.Tensor,
                   iters: int = 100) -> torch.Tensor:
    """Capture the vector-add launch in a CUDA graph and replay it `iters` times.

    Inputs are device tensors. Warmup compiles the kernel before capture (a graph
    cannot capture a compile); replay then removes the per-launch host overhead.

    Args:
        a: 1-D CUDA tensor.
        b: 1-D CUDA tensor, same shape and dtype as a.
        iters: Number of graph replays.

    Returns:
        CUDA tensor with the elementwise sum.
    """
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            vector_add(a, b)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = vector_add(a, b)
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return out
