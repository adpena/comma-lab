"""Tests for canonical top-submission reverse-engineering tooling."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reverse_engineer_top_submissions import (
    CURRENT_FLOOR_SPECS,
    PR_SPECS,
    _pr65_segments,
)


def test_top_submission_specs_are_pinned() -> None:
    specs = {spec.pr_number: spec for spec in PR_SPECS}
    assert set(specs) == {65, 67}
    for spec in specs.values():
        assert len(spec.expected_commit) == 40
        assert len(spec.expected_archive_sha256) == 64
        assert spec.expected_archive_bytes > 200_000
        assert spec.container_member in {"p", "x"}


def test_current_floor_specs_are_pinned() -> None:
    specs = {spec.pr_number: spec for spec in CURRENT_FLOOR_SPECS}
    assert set(specs) == {63, 64}
    for spec in specs.values():
        assert len(spec.expected_commit) == 40
        assert len(spec.expected_archive_sha256) == 64
        assert spec.expected_archive_bytes > 280_000
        assert spec.container_member == "p"
        assert spec.archive_url.startswith("https://github.com/")


def test_pr65_length_table_parser_is_little_endian_24bit() -> None:
    payload = (
        (219_472).to_bytes(3, "little")
        + (57_074).to_bytes(3, "little")
        + (1_487).to_bytes(3, "little")
        + b"payload"
    )
    parsed = _pr65_segments(payload)
    assert parsed["first_32_24bit_lengths"][:3] == [219_472, 57_074, 1_487]
