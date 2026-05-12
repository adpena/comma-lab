"""Tests for durable output-path policy helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.output_path_policy import assert_not_temporary_output_dir


def test_temporary_output_roots_are_refused() -> None:
    for path in ("/tmp/pact-out", "/var/tmp/pact-out", "/private/tmp/pact-out"):
        with pytest.raises(ValueError, match="no_tmp_paths"):
            assert_not_temporary_output_dir(Path(path), tool_name="unit")


def test_durable_output_dir_is_returned_resolved() -> None:
    resolved = assert_not_temporary_output_dir(
        Path("experiments/results/unit-output"),
        tool_name="unit",
    )
    assert resolved.is_absolute()
    assert resolved.name == "unit-output"
