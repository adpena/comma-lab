# SPDX-License-Identifier: MIT
"""Tests for shared substrate inflate runtime helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.substrates._shared.inflate_runtime import raw_output_path


def test_raw_output_path_preserves_safe_relative_subdirs(tmp_path: Path) -> None:
    assert raw_output_path(tmp_path, "0.mkv") == tmp_path / "0.raw"
    assert raw_output_path(tmp_path, "nested/0.mkv") == tmp_path / "nested" / "0.raw"


@pytest.mark.parametrize(
    "name",
    ["", "../0.mkv", "nested/../../0.mkv", "/tmp/0.mkv", "nested//0.mkv"],
)
def test_raw_output_path_rejects_escape_or_empty_entries(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(ValueError, match=r"unsafe|escapes"):
        raw_output_path(tmp_path, name)
