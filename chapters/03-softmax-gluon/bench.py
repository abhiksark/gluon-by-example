# chapters/03-softmax-gluon/bench.py
"""Benchmarks softmax head-to-head: torch vs Triton vs Gluon.

Fixed M=4096 rows, sweeping row width N. Writes CSV + chart.

Usage: python chapters/03-softmax-gluon/bench.py
"""

import csv
import sys
from pathlib import Path

import torch
import triton

from gluon_by_example.gluon_impl.softmax import softmax as gluon_softmax
from gluon_by_example.triton_impl.softmax import softmax as triton_softmax

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from bench_utils import gpu_slug  # noqa: E402
from make_chart import make_bandwidth_chart  # noqa: E402

M = 4096  # rows
COLS = [2**i for i in range(8, 15)]  # 256 .. 16384

PROVIDERS = {
    "torch": lambda x: torch.softmax(x, dim=-1),
    "triton": triton_softmax,
    "gluon": gluon_softmax,
}


def main() -> None:
    """Runs softmax benchmark across providers and row widths, writes CSV + PNG."""
    rows = []
    for n in COLS:
        x = torch.randn(M, n, device="cuda")
        for provider, fn in PROVIDERS.items():
            ms = triton.testing.do_bench(lambda: fn(x))
            # Ideal traffic: read MN + write MN elements; ms -> GB/s.
            gbps = 2 * x.numel() * x.element_size() / ms * 1e-6
            rows.append({"n": n, "provider": provider, "gbps": round(gbps, 2)})
            print(f"N={n:>6}  {provider:<8} {gbps:8.1f} GB/s")

    slug = gpu_slug()
    results_dir = REPO / "benchmarks" / "results"
    charts_dir = REPO / "benchmarks" / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"softmax-gluon-{slug}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "gbps"])
        writer.writeheader()
        writer.writerows(rows)

    png_path = charts_dir / f"softmax-gluon-{slug}.png"
    make_bandwidth_chart(
        csv_path,
        png_path,
        title=f"softmax: Triton vs Gluon, {M} rows — {torch.cuda.get_device_name(0)}",
        xlabel="columns per row",
    )
    print(f"wrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
