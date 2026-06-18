# chapters/09-normalization/bench.py
"""Benchmarks LayerNorm: torch vs Triton forward, plus atomic vs partial backward.

Fixed M=4096 rows, sweeping row width N. Writes CSV + chart.

The backward providers move roughly 3x the bytes of the forward (they read x,
dy, mean, rstd and write dx, dw, db), so their effective-bandwidth number is
not directly comparable to the forward rows. The backward comparison isolates
atomic-add vs two-stage-partial for the weight-gradient reduction only.

Usage: python chapters/09-normalization/bench.py
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from gluon_by_example.triton_impl.normalization import layer_norm as tn_layer_norm
from gluon_by_example.triton_impl.normalization import set_dw_mode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from bench_utils import gpu_slug  # noqa: E402
from make_chart import make_bandwidth_chart  # noqa: E402

M = 4096  # rows
COLS = [2**i for i in range(8, 15)]  # 256 .. 16384


def require_idle_gpu() -> None:
    """Exits if any other process is using the benchmark GPU.

    A shared card throttles clocks and steals SMs; bandwidth-bound numbers
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
            sys.exit(f"{msg} Benchmark refused: bandwidth-bound numbers "
                     "on a shared card are fiction.")


def main() -> None:
    """Runs LayerNorm benchmark across providers and row widths, writes CSV + PNG."""
    require_idle_gpu()
    rows = []
    for n in COLS:
        # Forward tensors (no grad needed for forward-only providers).
        x_fwd = torch.randn(M, n, device="cuda")
        w_fwd = torch.ones(n, device="cuda")
        b_fwd = torch.zeros(n, device="cuda")

        # Torch forward.
        ms = triton.testing.do_bench(
            lambda: F.layer_norm(x_fwd, (n,), w_fwd, b_fwd)
        )
        # Effective bandwidth: read M*N + write M*N, charged uniformly so
        # forward providers are directly comparable.
        gbps = 2 * M * n * x_fwd.element_size() / ms * 1e-6
        rows.append({"n": n, "provider": "torch", "gbps": round(gbps, 2)})
        print(f"N={n:>6}  {'torch':<20} {gbps:8.1f} GB/s")

        # Triton forward.
        ms = triton.testing.do_bench(
            lambda: tn_layer_norm(x_fwd, w_fwd, b_fwd)
        )
        gbps = 2 * M * n * x_fwd.element_size() / ms * 1e-6
        rows.append({"n": n, "provider": "triton", "gbps": round(gbps, 2)})
        print(f"N={n:>6}  {'triton':<20} {gbps:8.1f} GB/s")

        # Backward providers: tensors with requires_grad.
        x_bwd = torch.randn(M, n, device="cuda", requires_grad=True)
        w_bwd = torch.ones(n, device="cuda", requires_grad=True)
        b_bwd = torch.zeros(n, device="cuda", requires_grad=True)
        dy = torch.randn(M, n, device="cuda")

        # triton-dw-atomic: backward with atomic weight-grad reduction.
        set_dw_mode("atomic")

        def fwd_bwd_atomic():
            x_bwd.grad = w_bwd.grad = b_bwd.grad = None
            tn_layer_norm(x_bwd, w_bwd, b_bwd).backward(dy)

        ms = triton.testing.do_bench(fwd_bwd_atomic)
        # Backward moves ~3x the bytes of the forward; this number isolates
        # atomic-vs-partial and is not directly comparable to forward rows.
        gbps = 2 * M * n * x_bwd.element_size() / ms * 1e-6
        rows.append({"n": n, "provider": "triton-dw-atomic", "gbps": round(gbps, 2)})
        print(f"N={n:>6}  {'triton-dw-atomic':<20} {gbps:8.1f} GB/s")

        # triton-dw-partial: backward with two-stage partial reduction.
        set_dw_mode("partial")

        def fwd_bwd_partial():
            x_bwd.grad = w_bwd.grad = b_bwd.grad = None
            tn_layer_norm(x_bwd, w_bwd, b_bwd).backward(dy)

        ms = triton.testing.do_bench(fwd_bwd_partial)
        gbps = 2 * M * n * x_bwd.element_size() / ms * 1e-6
        rows.append({"n": n, "provider": "triton-dw-partial", "gbps": round(gbps, 2)})
        print(f"N={n:>6}  {'triton-dw-partial':<20} {gbps:8.1f} GB/s")

    slug = gpu_slug()
    results_dir = REPO / "benchmarks" / "results"
    charts_dir = REPO / "benchmarks" / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"normalization-{slug}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "gbps"])
        writer.writeheader()
        writer.writerows(rows)

    png_path = charts_dir / f"normalization-{slug}.png"
    make_bandwidth_chart(
        csv_path,
        png_path,
        title=f"LayerNorm fused-row, {M} rows -- {torch.cuda.get_device_name(0)}",
        xlabel="columns per row",
    )
    print(f"wrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
