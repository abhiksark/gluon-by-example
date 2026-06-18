# chapters/12-attention-gluon/bench.py
"""Benchmarks FlashAttention forward: torch SDPA vs Gluon FA.

Fixed Z=2, H=8, D=64; sweeps sequence length N in [512..8192]. Both causal
and non-causal variants are timed. Compute-bound kernels are sensitive to
contention, so the script refuses to run while any other process holds the
benchmark GPU.

The Gluon kernel is GPU-verified against torch SDPA / autograd; see the
module docstring in gluon_impl/attention.py for the SliceLayout discipline.

Usage: python chapters/12-attention-gluon/bench.py
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import gluon_by_example.gluon_impl.attention as ga

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from bench_utils import gpu_slug  # noqa: E402
from make_chart import make_bandwidth_chart  # noqa: E402

# Fixed outer dimensions; sweep the sequence length.
Z = 2
H = 8
D = 64
SEQ_LENS = [2**i for i in range(9, 14)]  # 512, 1024, 2048, 4096, 8192


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


def _tflops(n: int, ms: float) -> float:
    """Attention forward TFLOP/s.

    Full non-causal attention computes 4 * Z * H * N^2 * D FLOPs (two N x N
    matmuls, each costing 2 * Z * H * N * N * D multiply-adds). Causal masks
    roughly half the KV blocks, so the actual work is ~half; we still charge
    the full non-causal FLOPs as the denominator so both variants are on a
    comparable scale. The causal number will therefore read lower, reflecting
    the real hardware utilisation rather than an inflated "causal TFLOP/s."
    """
    flops = 4 * Z * H * n * n * D
    return flops / ms * 1e-9


def main() -> None:
    """Runs attention forward benchmark across providers and sequence lengths."""
    require_idle_gpu()

    rows = []
    for n in SEQ_LENS:
        q = torch.randn(Z, H, n, D, device="cuda", dtype=torch.float16)
        k = torch.randn(Z, H, n, D, device="cuda", dtype=torch.float16)
        v = torch.randn(Z, H, n, D, device="cuda", dtype=torch.float16)

        for causal in (False, True):
            suffix = "-causal" if causal else ""

            ms_torch = triton.testing.do_bench(
                lambda causal=causal: F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal
                )
            )
            rows.append({
                "n": n,
                "provider": f"torch{suffix}",
                "tflops": round(_tflops(n, ms_torch), 2),
            })
            print(f"N={n:>5}  {'torch'+suffix:<16}  {_tflops(n, ms_torch):8.1f} TFLOP/s")

            ms_gluon = triton.testing.do_bench(
                lambda causal=causal: ga.attention(q, k, v, causal=causal)
            )
            rows.append({
                "n": n,
                "provider": f"gluon{suffix}",
                "tflops": round(_tflops(n, ms_gluon), 2),
            })
            print(f"N={n:>5}  {'gluon'+suffix:<16}  {_tflops(n, ms_gluon):8.1f} TFLOP/s")

    slug = gpu_slug()
    results_dir = REPO / "benchmarks" / "results"
    charts_dir = REPO / "benchmarks" / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"attention-gluon-{slug}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "tflops"])
        writer.writeheader()
        writer.writerows(rows)

    png_path = charts_dir / f"attention-gluon-{slug}.png"
    make_bandwidth_chart(
        csv_path,
        png_path,
        title=f"FlashAttention forward, Z={Z} H={H} D={D} -- {torch.cuda.get_device_name(0)}",
        xlabel="sequence length N",
        ycol="tflops",
        ylabel="TFLOP/s",
    )
    print(f"wrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
