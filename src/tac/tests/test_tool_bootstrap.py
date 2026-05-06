from __future__ import annotations

import sys
from pathlib import Path

from tools.tool_bootstrap import prepend_paths


def test_prepend_paths_preserves_requested_order_without_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    original = list(sys.path)
    try:
        sys.path[:] = [str(second), "tail"]

        prepend_paths(first, second)

        assert sys.path[:3] == [str(first), str(second), "tail"]
        assert sys.path.count(str(first)) == 1
        assert sys.path.count(str(second)) == 1
    finally:
        sys.path[:] = original
