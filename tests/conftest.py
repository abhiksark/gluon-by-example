# tests/conftest.py
"""Shared pytest setup: make per-chapter demo modules importable by bare name."""

import sys
from pathlib import Path

# Some chapters ship a host-side demo module (not part of the installed
# gluon_by_example package) that its test imports directly. Put those dirs on
# the path so `from overlap_demo import ...` resolves.
_REPO = Path(__file__).resolve().parents[1]
for _chapter in ("08-overlap",):
    sys.path.insert(0, str(_REPO / "chapters" / _chapter))
