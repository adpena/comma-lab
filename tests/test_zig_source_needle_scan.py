from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools.zig_source_needle_scan import run_zig_source_scan


pytestmark = pytest.mark.skipif(shutil.which("zig") is None, reason="zig not installed")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _python_oracle(
    root: Path,
    *,
    dirs: tuple[str, ...],
    suffixes: tuple[str, ...],
    needles: tuple[str, ...],
    require_all: bool,
) -> list[str]:
    rows: list[str] = []
    for directory in dirs:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if not path.is_file():
                continue
            if (
                "__pycache__" in rel
                or rel.startswith(".git/")
                or rel.startswith(".mypy_cache/")
                or rel.startswith(".pytest_cache/")
                or rel.startswith(".omx/cache/")
                or "/.mypy_cache/" in rel
                or "/.pytest_cache/" in rel
                or "/.omx/cache/" in rel
                or rel.startswith("experiments/results/")
            ):
                continue
            if not any(rel.endswith(suffix) for suffix in suffixes):
                continue
            text = path.read_text(encoding="utf-8")
            hits = [needle for needle in needles if needle in text]
            if (len(hits) == len(needles)) if require_all else bool(hits):
                rows.append(rel)
    return rows


def test_zig_source_scan_matches_python_oracle_any(tmp_path: Path) -> None:
    _write(tmp_path / "src/tac/a.py", "eval_roundtrip = True\n")
    _write(tmp_path / "src/tac/b.py", "device = 'mps'\n")
    _write(tmp_path / "src/tac/c.txt", "mps eval_roundtrip\n")
    _write(tmp_path / "src/tac/__pycache__/ignored.py", "mps eval_roundtrip\n")
    _write(tmp_path / "experiments/results/ignored.py", "mps\n")
    _write(tmp_path / ".omx/cache/ignored.py", "mps\n")

    payload = run_zig_source_scan(
        root=tmp_path,
        dirs=("src/tac", "experiments"),
        suffixes=(".py",),
        needles=("mps", "eval_roundtrip"),
        require_all=False,
        binary_path=tmp_path / "source_needle_scan",
    )

    actual = [row["path"] for row in payload["matches"]]
    expected = _python_oracle(
        tmp_path,
        dirs=("src/tac", "experiments"),
        suffixes=(".py",),
        needles=("mps", "eval_roundtrip"),
        require_all=False,
    )
    assert actual == expected
    assert payload["match_count"] == 2


def test_zig_source_scan_matches_python_oracle_require_all(tmp_path: Path) -> None:
    _write(tmp_path / "tools/a.py", "lane_id = 'lane_demo'\n")
    _write(tmp_path / "tools/b.py", "lane_id = 'x'\nscore_claim = False\n")
    _write(tmp_path / "tools/c.sh", "lane_id score_claim\n")

    payload = run_zig_source_scan(
        root=tmp_path,
        dirs=("tools",),
        suffixes=(".py", ".sh"),
        needles=("lane_id", "score_claim"),
        require_all=True,
        binary_path=tmp_path / "source_needle_scan",
    )

    actual = [row["path"] for row in payload["matches"]]
    expected = _python_oracle(
        tmp_path,
        dirs=("tools",),
        suffixes=(".py", ".sh"),
        needles=("lane_id", "score_claim"),
        require_all=True,
    )
    assert actual == expected
    assert payload["matches"][0]["hit_count"] == 2
