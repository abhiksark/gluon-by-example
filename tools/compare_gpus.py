# tools/compare_gpus.py
"""Cross-GPU comparison charts from per-GPU benchmark CSVs.

Discovers benchmarks/results/<kernel>-<gpu-slug>.csv for a kernel, then
renders one figure with a subplot per provider and a line per GPU.

Usage: python tools/compare_gpus.py vector_add
       python tools/compare_gpus.py matmul --ycol tflops --ylabel TFLOP/s --xlabel "M = N = K"
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# One stable color per GPU so every chart in a report reads the same.
GPU_COLORS = {
    "nvidia-rtx-a6000": "#1f77b4",
    "nvidia-geforce-rtx-4090": "#ff7f0e",
    "nvidia-geforce-rtx-5090": "#2ca02c",
}


def gpu_label(slug: str) -> str:
    """Returns a human label for a GPU slug, e.g. ``RTX 4090``."""
    return slug.replace("nvidia-", "").replace("geforce-", "").replace("-", " ").upper()


def load_results(kernel: str, ycol: str) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Loads every per-GPU CSV for a kernel.

    Args:
        kernel: Kernel name prefix, e.g. ``vector_add``.
        ycol: Metric column to read.

    Returns:
        Mapping of provider -> gpu slug -> sorted (n, metric) points.
    """
    data: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    paths = sorted((REPO / "benchmarks" / "results").glob(f"{kernel}-*.csv"))
    if not paths:
        raise SystemExit(f"no CSVs found for kernel {kernel!r}")
    for path in paths:
        slug = path.stem[len(kernel) + 1:]
        with open(path) as f:
            for row in csv.DictReader(f):
                data[row["provider"]][slug].append((int(row["n"]), float(row[ycol])))
    for providers in data.values():
        for points in providers.values():
            points.sort()
    return data


def make_compare_chart(
    kernel: str,
    out_path: Path,
    xlabel: str = "elements",
    ycol: str = "gbps",
    ylabel: str = "GB/s",
) -> None:
    """Renders the per-provider, per-GPU comparison figure for one kernel.

    Args:
        kernel: Kernel name prefix, e.g. ``softmax``.
        out_path: Destination PNG path.
        xlabel: X-axis label.
        ycol: Metric column to plot.
        ylabel: Y-axis label.
    """
    data = load_results(kernel, ycol)
    providers = sorted(data)
    fig, axes = plt.subplots(
        1, len(providers), figsize=(5.5 * len(providers), 4.5), dpi=150,
        sharey=True, squeeze=False,
    )
    for ax, provider in zip(axes[0], providers):
        for slug, points in sorted(data[provider].items()):
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                linewidth=2,
                label=gpu_label(slug),
                color=GPU_COLORS.get(slug),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_title(provider)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend()
    fig.suptitle(f"{kernel} across GPUs")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("--xlabel", default="elements")
    parser.add_argument("--ycol", default="gbps")
    parser.add_argument("--ylabel", default="GB/s")
    args = parser.parse_args()
    out = REPO / "benchmarks" / "charts" / f"{args.kernel}-compare.png"
    make_compare_chart(args.kernel, out, args.xlabel, args.ycol, args.ylabel)


if __name__ == "__main__":
    main()
