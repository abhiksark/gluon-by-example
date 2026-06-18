# chapters/08-overlap/bench.py
"""Benchmarks host-side overlap on the book's vector-add kernel.

Two stories, two CSVs, two charts:
  1. Copy-compute overlap (overlap-<slug>.csv): serial vs multi-stream
     effective end-to-end GB/s as the chunk count grows, at a fixed large
     pinned input. Overlap hides the PCIe copies behind compute.
  2. CUDA graph replay (overlap-graph-<slug>.csv): per-launch time for eager
     launch vs graph replay across kernel sizes. The graph removes the
     per-launch host overhead, which dominates for small kernels.

Both effects distort badly under GPU contention, so the script refuses to run
while another process holds the benchmark GPU.

Usage: python chapters/08-overlap/bench.py
"""

import csv
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch
import triton

from overlap_demo import overlapped_add, serial_add

from gluon_by_example.triton_impl.vector_add import vector_add

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from bench_utils import gpu_slug  # noqa: E402
from make_chart import make_bandwidth_chart  # noqa: E402

# Copy-compute overlap: one large input, swept over chunk counts.
OVERLAP_N = 1 << 24  # 16M fp32 elements: 64MB per array, 192MB round-tripped
CHUNK_COUNTS = [1, 2, 4, 8, 16, 32, 64]

# Graph replay: a tiny kernel launched back to back, swept over launch count.
# A blocking loop (no CUDA-queue amortization) exposes the per-launch host cost
# that the graph removes; the kernel is small so that host cost dominates.
GRAPH_N = 1 << 12  # 4K element kernel: finishes instantly, so launch cost shows
GRAPH_COUNTS = [100, 200, 500, 1000, 2000, 5000, 10000]


def require_idle_gpu() -> None:
    """Exits if any other process is using the benchmark GPU.

    A shared card throttles clocks and steals SMs; overlap and launch-overhead
    numbers measured that way are fiction. Only the device the benchmark runs
    on is checked (first CUDA_VISIBLE_DEVICES entry, or GPU 0). Set
    GBE_ALLOW_SHARED=1 to downgrade the refusal to a warning.
    """
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    uuid_by_index = dict(line.split(", ") for line in gpus)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    target = uuid_by_index[visible.split(",")[0] if visible else "0"]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    others = [
        pid for pid, uuid in (line.split(", ") for line in apps if line)
        if uuid == target and int(pid) != os.getpid()
    ]
    if others:
        msg = f"benchmark GPU is busy (other compute PIDs: {', '.join(others)})."
        if os.environ.get("GBE_ALLOW_SHARED") == "1":
            print(f"WARNING: {msg} Proceeding because GBE_ALLOW_SHARED=1; "
                  "treat these numbers as indicative, not authoritative.",
                  file=sys.stderr)
        else:
            sys.exit(f"{msg} Benchmark refused: overlap numbers on a shared "
                     "card are fiction.")


def _overlap_rows() -> list[dict]:
    """Effective end-to-end GB/s for serial vs overlapped, vs chunk count."""
    a = torch.randn(OVERLAP_N, pin_memory=True)
    b = torch.randn(OVERLAP_N, pin_memory=True)
    bytes_moved = 3 * OVERLAP_N * a.element_size()  # a + b in, out back

    ms_serial = triton.testing.do_bench(lambda: serial_add(a, b))
    gbps_serial = bytes_moved / ms_serial / 1e6

    rows = []
    for chunks in CHUNK_COUNTS:
        ms = triton.testing.do_bench(lambda chunks=chunks: overlapped_add(a, b, chunks))
        gbps = bytes_moved / ms / 1e6
        rows.append({"n": chunks, "provider": "serial", "gbps": round(gbps_serial, 1)})
        rows.append({"n": chunks, "provider": "overlap", "gbps": round(gbps, 1)})
        print(f"chunks={chunks:>3}  serial {gbps_serial:7.1f}  overlap {gbps:7.1f} GB/s")
    return rows


def _time_us_per_launch(call, count: int) -> float:
    """Wall-clock microseconds per launch for `count` back-to-back blocking calls."""
    torch.cuda.synchronize()
    t0 = perf_counter()
    for _ in range(count):
        call()
    torch.cuda.synchronize()
    return (perf_counter() - t0) / count * 1e6


def _graph_rows() -> list[dict]:
    """Per-launch microseconds: eager launch vs graph replay, vs launch count.

    One fixed tiny kernel, launched `count` times in a blocking loop. Eager
    pays the host launch cost every time; the graph replays a captured sequence
    and removes it. The gap between the two lines is the per-launch host
    overhead the graph eliminates.
    """
    a = torch.randn(GRAPH_N, device="cuda")
    b = torch.randn(GRAPH_N, device="cuda")

    # Warmup compiles the kernel (a graph cannot capture a compile), then capture.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            vector_add(a, b)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        vector_add(a, b)

    rows = []
    for count in GRAPH_COUNTS:
        us_eager = _time_us_per_launch(lambda: vector_add(a, b), count)
        us_graph = _time_us_per_launch(graph.replay, count)
        rows.append({"n": count, "provider": "eager", "us": round(us_eager, 3)})
        rows.append({"n": count, "provider": "graph", "us": round(us_graph, 3)})
        print(f"count={count:>6}  eager {us_eager:6.2f}  graph {us_graph:6.2f} us/launch")
    return rows


def _write(rows: list[dict], stem: str, fieldnames: list[str], *,
           title: str, xlabel: str, ycol: str, ylabel: str) -> None:
    slug = gpu_slug()
    results_dir = REPO / "benchmarks" / "results"
    charts_dir = REPO / "benchmarks" / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{stem}-{slug}.csv"
    png_path = charts_dir / f"{stem}-{slug}.png"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    make_bandwidth_chart(
        csv_path, png_path,
        title=f"{title} -- {torch.cuda.get_device_name(0)}",
        xlabel=xlabel, ycol=ycol, ylabel=ylabel,
    )
    print(f"wrote {csv_path}\nwrote {png_path}")


def main() -> None:
    """Runs both overlap benchmarks and writes their CSVs and charts."""
    require_idle_gpu()
    _write(_overlap_rows(), "overlap", ["n", "provider", "gbps"],
           title="Copy-compute overlap (vector-add, 16M elements)",
           xlabel="chunks (streams)", ycol="gbps", ylabel="effective GB/s")
    _write(_graph_rows(), "overlap-graph", ["n", "provider", "us"],
           title="CUDA graph replay vs eager launch (4K vector-add)",
           xlabel="launch count", ycol="us", ylabel="us per launch")


if __name__ == "__main__":
    main()
