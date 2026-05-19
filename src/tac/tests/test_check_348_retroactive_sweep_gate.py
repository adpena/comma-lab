# SPDX-License-Identifier: MIT
"""Catalog #348 retroactive sweep evidence gate tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_new_gate_landing_includes_retroactive_sweep_evidence,
)


def _git_init(repo_root: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo_root, check=True)


def _write(repo_root: Path, rel: str, text: str) -> None:
    path = repo_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo_root: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo_root, check=True)


def _write_base_repo(repo_root: Path) -> None:
    _git_init(repo_root)
    _write(repo_root, "src/tac/preflight.py", "# base preflight\n")
    _write(repo_root, "CLAUDE.md", "# Catalog\n")
    _commit(repo_root, "base")


def _add_gate_commit(
    repo_root: Path,
    *,
    function_name: str = "check_example_new_gate",
    catalog_number: int | None = 999,
    waiver: str | None = None,
    multiline: bool = False,
) -> None:
    suffix = f"  # RETROACTIVE_SWEEP_WAIVED:{waiver}" if waiver is not None else ""
    if multiline:
        signature = (
            f"def {function_name}(\n"
            "    *,\n"
            "    strict: bool = False,\n"
            "    verbose: bool = False,\n"
            f") -> list[str]:{suffix}\n"
            "    return []\n"
        )
    else:
        signature = (
            f"def {function_name}(*, strict=False, verbose=False):{suffix}\n"
            "    return []\n"
        )
    _write(
        repo_root,
        "src/tac/preflight.py",
        f"# base preflight\n\n{signature}",
    )
    if catalog_number is not None:
        _write(
            repo_root,
            "CLAUDE.md",
            f"# Catalog\n\n{catalog_number}. `{function_name}` - synthetic gate row.\n",
        )
    _commit(repo_root, f"add {function_name}")


def _write_uncommitted_gate(
    repo_root: Path,
    *,
    function_name: str = "check_example_new_gate",
    catalog_number: int | None = 999,
) -> None:
    _write(
        repo_root,
        "src/tac/preflight.py",
        f"# base preflight\n\ndef {function_name}(*, strict=False, verbose=False):\n    return []\n",
    )
    if catalog_number is not None:
        _write(
            repo_root,
            "CLAUDE.md",
            f"# Catalog\n\n{catalog_number}. `{function_name}` - synthetic gate row.\n",
        )


def _write_complete_sweep(repo_root: Path, catalog_number: int = 999) -> None:
    _write(
        repo_root,
        f".omx/research/retroactive_sweep_for_catalog_{catalog_number}_20260519T000000Z.md",
        """# Retroactive Sweep

## Bug-class symptom signature
Synthetic stale verdict class.

## Gate Identity
Catalog #999, `check_example_new_gate`.

## Search Command
`git log -p -- src/tac/preflight.py`

## Pre-fix window
base..HEAD

## Historical-KILL/DEFER/FALSIFY search results
No affected historical verdicts in synthetic fixture.

## Per-finding RE-EVAL-priority assignment
No re-eval needed.
""",
    )


def test_clean_repo_without_new_gate_passes(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)

    assert check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    ) == []


def test_new_gate_without_sweep_memo_is_reported(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(tmp_path)

    violations = check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    )

    assert len(violations) == 1
    assert "Catalog #999 missing" in violations[0]
    assert "retroactive_sweep_for_catalog_999" in violations[0]


def test_new_gate_with_complete_sweep_memo_passes(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(tmp_path)
    _write_complete_sweep(tmp_path)

    assert check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    ) == []


def test_new_gate_with_incomplete_sweep_memo_reports_missing_fields(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(tmp_path)
    _write(
        tmp_path,
        ".omx/research/retroactive_sweep_for_catalog_999_20260519T000000Z.md",
        "## Bug-class symptom signature\nonly one field\n",
    )

    violations = check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    )

    assert len(violations) == 1
    assert "lacks the 4-field contract" in violations[0]
    assert "pre-fix window" in violations[0]


def test_new_gate_with_real_same_line_waiver_passes(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(
        tmp_path,
        catalog_number=None,
        waiver="legacy_gate_has_no_historical_verdict_surface",
    )

    assert check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    ) == []


def test_new_multiline_gate_with_closing_line_waiver_passes(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(
        tmp_path,
        catalog_number=None,
        waiver="legacy_gate_has_no_historical_verdict_surface",
        multiline=True,
    )

    assert check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    ) == []


def test_unstaged_worktree_new_gate_is_reported_before_commit(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _write_uncommitted_gate(tmp_path)

    violations = check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=1,
    )

    assert len(violations) == 1
    assert violations[0].startswith("WORKTREE:check_example_new_gate")


def test_staged_new_gate_is_reported_before_commit(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _write_uncommitted_gate(tmp_path)
    subprocess.run(["git", "add", "src/tac/preflight.py", "CLAUDE.md"], cwd=tmp_path, check=True)

    violations = check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=1,
    )

    assert len(violations) == 1
    assert violations[0].startswith("INDEX:check_example_new_gate")


def test_new_gate_placeholder_waiver_is_rejected(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(tmp_path, catalog_number=None, waiver="<rationale>")

    violations = check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    )

    assert len(violations) == 1
    assert "invalid RETROACTIVE_SWEEP_WAIVED rationale" in violations[0]


def test_new_gate_without_catalog_row_is_reported(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(tmp_path, catalog_number=None)

    violations = check_new_gate_landing_includes_retroactive_sweep_evidence(
        repo_root=tmp_path,
        recent_commits=5,
    )

    assert len(violations) == 1
    assert "no CLAUDE.md catalog row" in violations[0]


def test_strict_mode_raises(tmp_path: Path) -> None:
    _write_base_repo(tmp_path)
    _add_gate_commit(tmp_path)

    with pytest.raises(PreflightError, match="Catalog #348"):
        check_new_gate_landing_includes_retroactive_sweep_evidence(
            repo_root=tmp_path,
            recent_commits=5,
            strict=True,
        )


def test_live_orchestrator_wires_catalog_348_warn_only() -> None:
    text = Path("src/tac/preflight.py").read_text(encoding="utf-8")
    call = "check_new_gate_landing_includes_retroactive_sweep_evidence("
    assert call in text
    idx = text.index(call)
    assert "strict=False" in text[idx : idx + 160]


def test_live_recent_gate_backfill_count_is_bounded() -> None:
    violations = check_new_gate_landing_includes_retroactive_sweep_evidence(
        recent_commits=20,
    )
    assert isinstance(violations, list)
    assert len(violations) <= 30
