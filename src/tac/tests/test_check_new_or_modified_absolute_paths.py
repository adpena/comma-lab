"""Tests for the Catalog #208 staged absolute-path scope extension."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    changed_absolute_path_violations_for_text,
    check_no_new_or_modified_absolute_local_paths,
)


def test_text_detector_refuses_user_home_and_volume_literals() -> None:
    text = "home=/Users/example/work\ntier=/Volumes/Data/work\n"  # ABSOLUTE_PATH_OK:guard-positive-control-fixture
    violations = changed_absolute_path_violations_for_text("tool.py", text)
    assert len(violations) == 2
    assert all("local absolute path" in violation for violation in violations)


def test_text_detector_allows_portable_tokens() -> None:
    text = "home=~/work\ntier=$PACT_TIER1/work\nfallback=$PACT_TIER2/work\n"
    assert changed_absolute_path_violations_for_text("tool.py", text) == []


def test_specific_same_line_waiver_is_allowed() -> None:
    text = (
        "DEFAULT = '/Volumes/Data/work'  "  # ABSOLUTE_PATH_OK:guard-waiver-fixture
        "# ABSOLUTE_PATH_OK:runtime-resolver-default\n"
    )
    assert changed_absolute_path_violations_for_text("resolver.py", text) == []


@pytest.mark.parametrize(
    "rationale",
    ["<rationale>", "reason", "TODO", "TODO-fill-this-in", "intentional"],
)
def test_placeholder_waivers_are_refused(rationale: str) -> None:
    text = f"DEFAULT = '/Volumes/Data/work'  # ABSOLUTE_PATH_OK:{rationale}\n"  # ABSOLUTE_PATH_OK:guard-waiver-fixture
    assert changed_absolute_path_violations_for_text("resolver.py", text)


def test_default_mode_reads_staged_blob_not_clean_worktree(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    source = tmp_path / "example.py"
    source.write_text(
        "ROOT = '/Users/example/work'\n",  # ABSOLUTE_PATH_OK:guard-positive-control-fixture
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "example.py"], cwd=tmp_path, check=True)
    source.write_text("ROOT = '$HOME/work'\n", encoding="utf-8")

    violations = check_no_new_or_modified_absolute_local_paths(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert violations[0].startswith("example.py:1:")


def test_explicit_candidate_control_raises_strict(tmp_path: Path) -> None:
    source = tmp_path / "new_tool.py"
    source.write_text(
        "ROOT = '/Volumes/Private/work'\n",  # ABSOLUTE_PATH_OK:guard-positive-control-fixture
        encoding="utf-8",
    )
    with pytest.raises(PreflightError):
        check_no_new_or_modified_absolute_local_paths(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
            candidate_paths=["new_tool.py"],
        )
