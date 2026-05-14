# SPDX-License-Identifier: MIT
"""Catalog #166: Modal dispatch HEAD-parity ledger tests.

PHASE-B1-PIVOT bug-class anchor (2026-05-12). Two consecutive Modal A100
dispatches of ``experiments/train_substrate_sane_hnerv.py`` crashed rc=1
because the operator dispatched at 20:26:47Z BEFORE the FIX-H Part 1
(commit 6048d690) landed at 20:44:00Z. The Modal worker faithfully ran the
pre-fix code that was on local disk at dispatch time. Post-mortem could not
distinguish "Modal worker mounted stale code" from "operator dispatched
before fix landed" because ``modal_metadata.json`` did not record the
dispatch-time HEAD SHA OR the working-tree-dirty state.

Catalog #166 refuses any state of ``experiments/modal_train_lane.py`` that
drops the dispatch-time HEAD parity ledger or the worker-side HEAD parity
assertion.

Coverage targets:

- live count vs current repo state (must be 0)
- positive: missing _git_dirty_tree_summary helper -> violation
- positive: missing metadata_schema marker -> violation
- positive: missing CATALOG_166 worker-side warn token -> violation
- positive: missing modal_worker_head_ledger.json write -> violation
- positive: missing require_clean_head kwarg -> violation
- positive: missing sentinel_files_local_sha256 metadata key -> violation
- negative: every required surface present -> no violation
- file missing entirely -> violation
- strict mode raises PreflightError with full diagnostic
- non-strict returns the violation list
- verbose=True prints the diagnostic banner
- per-surface error message names the missing surface
- accepts a custom repo_root for test scaffolds
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_dispatch_verifies_worker_source_matches_head,
)

# Use this constant in fixtures so we don't have to re-import the private one
_REQUIRED_SURFACES = (
    "def _git_dirty_tree_summary",
    "modal_train_lane_dispatch_metadata_v2_catalog166",
    "mounted_code_git_head",
    "working_tree_dirty",
    "working_tree_dirty_summary",
    "sentinel_files_local_sha256",
    "worker_sentinel_sha256",
    "sentinel_mismatches",
    "modal_worker_head_ledger.json",
    "CATALOG_166",
    "require_clean_head",
)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """Build a minimal repo tree with a fully-compliant dispatcher file."""
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    body = "# fake dispatcher used by Catalog #166 tests\n"
    body += "\n".join(f"# surface: {s}" for s in _REQUIRED_SURFACES)
    body += "\n"
    (repo / "experiments" / "modal_train_lane.py").write_text(body)
    return repo


def test_live_repo_count_is_zero() -> None:
    """The current repo state MUST satisfy Catalog #166. If this fails, a
    sibling subagent edited the dispatcher and dropped a contract surface."""
    # This test runs against the actual repo root (resolved via cwd).
    violations = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=Path.cwd(), strict=False
    )
    assert violations == [], f"Catalog #166 live count != 0: {violations}"


def test_compliant_dispatcher_has_no_violations(fake_repo: Path) -> None:
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert out == []


def test_missing_dispatcher_file_is_violation(tmp_path: Path) -> None:
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=repo, strict=False
    )
    assert any("missing" in v for v in out)


def test_missing_dirty_tree_helper_is_violation(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    text = target.read_text()
    text = text.replace("def _git_dirty_tree_summary", "# stripped")
    target.write_text(text)
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert any("_git_dirty_tree_summary" in v for v in out)


def test_missing_metadata_schema_marker_is_violation(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    text = target.read_text()
    text = text.replace(
        "modal_train_lane_dispatch_metadata_v2_catalog166", "# stripped"
    )
    target.write_text(text)
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert any("metadata_v2_catalog166" in v for v in out)


def test_missing_catalog_166_worker_warn_is_violation(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    text = target.read_text()
    text = text.replace("CATALOG_166", "# stripped")
    target.write_text(text)
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert any("CATALOG_166" in v for v in out)


def test_missing_worker_head_ledger_write_is_violation(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    text = target.read_text()
    text = text.replace("modal_worker_head_ledger.json", "# stripped")
    target.write_text(text)
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert any("modal_worker_head_ledger" in v for v in out)


def test_missing_require_clean_head_is_violation(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    text = target.read_text()
    text = text.replace("require_clean_head", "# stripped")
    target.write_text(text)
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert any("require_clean_head" in v for v in out)


def test_missing_sentinel_files_metadata_key_is_violation(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    text = target.read_text()
    text = text.replace("sentinel_files_local_sha256", "# stripped")
    target.write_text(text)
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert any("sentinel_files_local_sha256" in v for v in out)


def test_strict_mode_raises_preflight_error(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    target.write_text("empty_dispatcher_no_surfaces\n")
    with pytest.raises(PreflightError) as excinfo:
        check_modal_dispatch_verifies_worker_source_matches_head(
            repo_root=fake_repo, strict=True
        )
    assert "Catalog #166" in str(excinfo.value)
    assert "PHASE-B1-PIVOT" in str(excinfo.value)


def test_non_strict_returns_violation_list(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    target.write_text("empty\n")
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    assert isinstance(out, list)
    assert len(out) >= 1


def test_verbose_prints_status_banner(fake_repo: Path, capsys) -> None:
    check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "modal-head-parity" in captured.out


def test_verbose_with_violations_prints_count(fake_repo: Path, capsys) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    target.write_text("empty\n")
    check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "modal-head-parity" in captured.out
    assert "violation" in captured.out


def test_per_surface_error_names_the_missing_surface(fake_repo: Path) -> None:
    target = fake_repo / "experiments" / "modal_train_lane.py"
    target.write_text("only_garbage\n")
    out = check_modal_dispatch_verifies_worker_source_matches_head(
        repo_root=fake_repo, strict=False
    )
    # Each missing surface should produce a violation row that names it.
    for surface in _REQUIRED_SURFACES:
        assert any(surface in v for v in out), f"missing per-surface error for {surface}"


def test_default_repo_root_is_cwd(fake_repo: Path, monkeypatch) -> None:
    """When repo_root is None the check uses Path.cwd()."""
    monkeypatch.chdir(fake_repo)
    out = check_modal_dispatch_verifies_worker_source_matches_head(strict=False)
    assert out == []
