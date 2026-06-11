# tests/test_make_chart.py
"""Tests for the shared benchmark chart generator."""

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from make_chart import make_bandwidth_chart  # noqa: E402


def test_writes_png_from_results_csv(tmp_path):
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "gbps"])
        writer.writeheader()
        for n in (4096, 65536):
            writer.writerow({"n": n, "provider": "triton", "gbps": 100.0})
            writer.writerow({"n": n, "provider": "gluon", "gbps": 110.0})

    out_path = tmp_path / "chart.png"
    make_bandwidth_chart(csv_path, out_path, title="test chart")

    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # a real PNG, not an empty file


def test_accepts_custom_xlabel(tmp_path):
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "gbps"])
        writer.writeheader()
        writer.writerow({"n": 1024, "provider": "triton", "gbps": 100.0})

    out_path = tmp_path / "chart.png"
    make_bandwidth_chart(csv_path, out_path, title="test", xlabel="columns")

    assert out_path.exists()


def test_accepts_custom_metric_column(tmp_path):
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "provider", "tflops"])
        writer.writeheader()
        writer.writerow({"n": 1024, "provider": "triton", "tflops": 50.0})

    out_path = tmp_path / "chart.png"
    make_bandwidth_chart(
        csv_path, out_path, title="test", ycol="tflops", ylabel="TFLOP/s"
    )

    assert out_path.exists()
