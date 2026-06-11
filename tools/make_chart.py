# tools/make_chart.py
"""Benchmark chart generator. The one chart style for the whole repo.

Reads a results CSV with columns: n, provider, and a metric column (default gbps). Writes a PNG.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_bandwidth_chart(
    csv_path: Path,
    out_path: Path,
    title: str,
    xlabel: str = "elements",
    ycol: str = "gbps",
    ylabel: str = "GB/s",
) -> None:
    """Renders a bandwidth-vs-size line chart from a results CSV.

    Args:
        csv_path: CSV with columns n (int), provider (str), and a metric column (default gbps).
        out_path: Destination PNG path.
        title: Chart title (kernel name + GPU).
        xlabel: X-axis label; defaults to "elements".
        ycol: Name of the metric column to plot.
        ylabel: Y-axis label.
    """
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            series[row["provider"]].append((int(row["n"]), float(row[ycol])))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for provider in sorted(series):
        points = sorted(series[provider])
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            linewidth=2,
            label=provider,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--xlabel", default="elements")
    parser.add_argument("--ycol", default="gbps")
    parser.add_argument("--ylabel", default="GB/s")
    args = parser.parse_args()
    make_bandwidth_chart(
        args.csv_path, args.out_path, args.title, args.xlabel, args.ycol, args.ylabel
    )


if __name__ == "__main__":
    main()
