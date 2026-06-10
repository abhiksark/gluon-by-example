# chapters/01-vector-add/bench.py
"""Benchmarks vector_add: torch vs Triton vs Gluon. Writes CSV + chart.

Usage: python chapters/01-vector-add/bench.py
"""

import csv
import re
import sys
from pathlib import Path

import torch
import triton

from gluon_by_example.gluon_impl.vector_add import vector_add as gluon_add
from gluon_by_example.triton_impl.vector_add import vector_add as triton_add

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from make_chart import make_bandwidth_chart  # noqa: E402

PROVIDERS = {
    "torch": torch.add,
    "triton": triton_add,
    "gluon": gluon_add,
}
SIZES = [2**i for i in range(12, 28, 2)]  # 4 Ki .. 64 Mi elements


def gpu_slug() -> str:
    """Returns a filesystem-safe slug derived from the CUDA device name.

    Returns:
        Lowercase alphanumeric slug with hyphens replacing non-alphanumeric
        characters, e.g. ``nvidia-rtx-a6000``.
    """
    name = torch.cuda.get_device_name(0).lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def main() -> None:
    """Runs vector-add benchmark across providers and sizes, writes CSV + PNG."""
    rows = []
    for n in SIZES:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        for provider, fn in PROVIDERS.items():
            ms = triton.testing.do_bench(lambda: fn(x, y))
            # 2 reads + 1 write of n fp32 elements; ms -> GB/s.
            gbps = 3 * n * x.element_size() / ms * 1e-6
            rows.append({"n": n, "provider": provider, "gbps": round(gbps, 2)})
            print(f"n={n:>10}  {provider:<8} {gbps:8.1f} GB/s")

    slug = gpu_slug()
    results_dir = REPO / "benchmarks" / "results"
    charts_dir = REPO / "benchmarks" / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"vector_add-{slug}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "gbps"])
        writer.writeheader()
        writer.writerows(rows)

    png_path = charts_dir / f"vector_add-{slug}.png"
    make_bandwidth_chart(
        csv_path, png_path, title=f"vector add — {torch.cuda.get_device_name(0)}"
    )
    print(f"wrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
