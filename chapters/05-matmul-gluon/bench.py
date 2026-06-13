# chapters/05-matmul-gluon/bench.py
"""Benchmarks matmul: Triton vs Gluon (floor + cp.async) vs cuBLAS, fp16 squares.

Compute-bound kernels are sensitive to contention and thermals, so this script
refuses to run while other processes hold the benchmark GPU. All sizes are
divisible by both Gluon tiles (the floor's K=64 and the pipelined kernel's K=32).

Usage: python chapters/05-matmul-gluon/bench.py
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton

from gluon_by_example.gluon_impl.matmul import matmul as gluon_matmul
from gluon_by_example.gluon_impl.matmul_pipelined import matmul as gluon_pipe_matmul
from gluon_by_example.triton_impl.matmul import matmul as triton_matmul

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from bench_utils import gpu_slug  # noqa: E402
from make_chart import make_bandwidth_chart  # noqa: E402

SIZES = [256, 512, 1024, 2048, 4096, 8192]  # M = N = K, all tile-aligned

PROVIDERS = {
    "torch": torch.matmul,
    "triton": triton_matmul,
    "gluon": gluon_matmul,
    "gluon-pipe": gluon_pipe_matmul,
}


def require_idle_gpu() -> None:
    """Exits if any other process is using the benchmark GPU.

    A shared card throttles clocks and steals SMs; compute-bound numbers
    measured that way are fiction. Only the device the benchmark runs on is
    checked (first CUDA_VISIBLE_DEVICES entry, or GPU 0). Set GBE_ALLOW_SHARED=1
    to downgrade the refusal to a warning.
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
            sys.exit(f"{msg} Benchmark refused: compute-bound numbers "
                     "on a shared card are fiction.")


def main() -> None:
    """Runs the matmul benchmark across providers and sizes, writes CSV + PNG."""
    require_idle_gpu()
    rows = []
    for s in SIZES:
        a = torch.randn(s, s, device="cuda", dtype=torch.float16)
        b = torch.randn(s, s, device="cuda", dtype=torch.float16)
        for provider, fn in PROVIDERS.items():
            ms = triton.testing.do_bench(lambda fn=fn: fn(a, b))
            tflops = 2 * s**3 / ms * 1e-9
            rows.append({"n": s, "provider": provider, "tflops": round(tflops, 2)})
            print(f"M=N=K={s:>5}  {provider:<8} {tflops:8.1f} TFLOP/s")

    slug = gpu_slug()
    results_dir = REPO / "benchmarks" / "results"
    charts_dir = REPO / "benchmarks" / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"matmul-gluon-{slug}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "tflops"])
        writer.writeheader()
        writer.writerows(rows)

    png_path = charts_dir / f"matmul-gluon-{slug}.png"
    make_bandwidth_chart(
        csv_path,
        png_path,
        title=f"matmul fp16: Triton vs Gluon (floor + cp.async) vs cuBLAS — {torch.cuda.get_device_name(0)}",
        xlabel="M = N = K",
        ycol="tflops",
        ylabel="TFLOP/s",
    )
    print(f"wrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
