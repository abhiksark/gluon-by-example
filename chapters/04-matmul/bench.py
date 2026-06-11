# chapters/04-matmul/bench.py
"""Benchmarks matmul: Triton vs cuBLAS (torch.matmul), fp16 square gemms.

Compute-bound kernels are exquisitely sensitive to contention and thermals,
so this script refuses to run while other processes hold the GPU.

Usage: python chapters/04-matmul/bench.py
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton

from gluon_by_example.triton_impl.matmul import matmul as triton_matmul

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from bench_utils import gpu_slug  # noqa: E402
from make_chart import make_bandwidth_chart  # noqa: E402

SIZES = [256, 512, 1024, 2048, 4096, 8192]  # M = N = K

PROVIDERS = {
    "torch": torch.matmul,
    "triton": triton_matmul,
}


def require_idle_gpu() -> None:
    """Exits if any other process is using the GPU.

    A shared card throttles clocks and steals SMs; compute-bound numbers
    measured that way are fiction, not benchmarks.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    others = [pid for pid in out if pid and int(pid) != os.getpid()]
    if others:
        sys.exit(
            f"GPU is busy (other compute PIDs: {', '.join(others)}). "
            "Benchmark refused: compute-bound numbers on a shared card are fiction."
        )


def main() -> None:
    """Runs the matmul benchmark across providers and sizes, writes CSV + PNG."""
    require_idle_gpu()
    rows = []
    for s in SIZES:
        a = torch.randn(s, s, device="cuda", dtype=torch.float16)
        b = torch.randn(s, s, device="cuda", dtype=torch.float16)
        for provider, fn in PROVIDERS.items():
            ms = triton.testing.do_bench(lambda: fn(a, b))
            # 2*M*N*K flops for a square gemm; ms -> TFLOP/s.
            tflops = 2 * s**3 / ms * 1e-9
            rows.append({"n": s, "provider": provider, "tflops": round(tflops, 2)})
            print(f"M=N=K={s:>5}  {provider:<8} {tflops:8.1f} TFLOP/s")

    slug = gpu_slug()
    results_dir = REPO / "benchmarks" / "results"
    charts_dir = REPO / "benchmarks" / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"matmul-{slug}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "tflops"])
        writer.writeheader()
        writer.writerows(rows)

    png_path = charts_dir / f"matmul-{slug}.png"
    make_bandwidth_chart(
        csv_path,
        png_path,
        title=f"matmul fp16: Triton vs cuBLAS — {torch.cuda.get_device_name(0)}",
        xlabel="M = N = K",
        ycol="tflops",
        ylabel="TFLOP/s",
    )
    print(f"wrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
