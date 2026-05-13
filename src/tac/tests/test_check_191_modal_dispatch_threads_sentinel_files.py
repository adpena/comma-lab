"""Tests for Catalog #191 — check_modal_dispatch_threads_sentinel_files_per_catalog_166.

SIREN PRE-DISPATCH AUDIT 2026-05-13 CRITICAL #2 self-protect.

Bug class: ``tools/operator_authorize.py::_dispatch_modal`` builds the
modal CLI cmd. Catalog #166's fail-closed sentinel-mismatch protection
(rc=13 at modal_train_lane.py:327-342) fires ONLY when
``--sentinel-files`` is non-empty, ``--require-clean-head`` is set, and
``--lane-id`` is threaded from the operator-approved recipe. Removing either
stale-code flag silently disables Catalog #166's protection. Removing
``--lane-id`` recreates split custody: the operator claim and Modal direct claim
can use different lane ids for the same paid GPU job.

Sister of Catalog #166 (HEAD-parity ledger surface).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_dispatch_threads_sentinel_files_per_catalog_166,
)

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _write_dispatcher(tmp_path: Path, content: str) -> Path:
    """Write a fake dispatcher at the canonical path."""
    target = tmp_path / "tools" / "operator_authorize.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Positive cases — gate should flag
# ─────────────────────────────────────────────────────────────────────────


def test_missing_dispatcher_is_violation(tmp_path):
    """If the canonical dispatcher file is missing, gate flags it."""
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(vs) == 1
    assert "missing" in vs[0]


def test_missing_sentinel_files_flag_is_violation(tmp_path):
    """Dispatcher without `--sentinel-files` flag is a violation."""
    root = _write_dispatcher(tmp_path,
        "def _dispatch_modal(...):\n"
        "    cmd = ['modal', 'run', '--detach', '--require-clean-head']\n"
        "    return subprocess.call(cmd)\n"
        "def _modal_sentinel_files(): return ''\n"
    )
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=root, strict=False, verbose=False,
    )
    assert any("--sentinel-files" in v for v in vs)


def test_missing_require_clean_head_flag_is_violation(tmp_path):
    """Dispatcher without `--require-clean-head` flag is a violation."""
    root = _write_dispatcher(tmp_path,
        "def _dispatch_modal(...):\n"
        "    cmd = ['modal', 'run', '--detach', '--sentinel-files', files]\n"
        "    return subprocess.call(cmd)\n"
        "def _modal_sentinel_files(): return ''\n"
    )
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=root, strict=False, verbose=False,
    )
    assert any("--require-clean-head" in v for v in vs)


def test_missing_helper_function_is_violation(tmp_path):
    """Dispatcher without the `_modal_sentinel_files` helper is flagged."""
    root = _write_dispatcher(tmp_path,
        "def _dispatch_modal(...):\n"
        "    cmd = ['--sentinel-files', '...', '--require-clean-head']\n"
    )
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=root, strict=False, verbose=False,
    )
    assert any("_modal_sentinel_files" in v for v in vs)


def test_missing_lane_id_flag_is_violation(tmp_path):
    """Dispatcher without `--lane-id` can split operator and Modal custody."""
    root = _write_dispatcher(tmp_path,
        "def _dispatch_modal(...):\n"
        "    cmd = ['--sentinel-files', files, '--require-clean-head']\n"
        "    return subprocess.call(cmd)\n"
        "def _modal_sentinel_files(): return ''\n"
    )
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=root, strict=False, verbose=False,
    )
    assert any("--lane-id" in v for v in vs)


def test_all_four_surfaces_missing_yields_four_violations(tmp_path):
    """All four contract surfaces flagged independently."""
    root = _write_dispatcher(tmp_path,
        "def _dispatch_modal(...):\n"
        "    cmd = ['modal', 'run']\n"
        "    return subprocess.call(cmd)\n"
    )
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=root, strict=False, verbose=False,
    )
    assert len(vs) == 4


# ─────────────────────────────────────────────────────────────────────────
# Negative cases — gate should accept
# ─────────────────────────────────────────────────────────────────────────


def test_all_four_surfaces_present_is_accepted(tmp_path):
    """Canonical dispatcher with all surfaces present is OK."""
    root = _write_dispatcher(tmp_path,
        "def _modal_sentinel_files(recipe):\n"
        "    return 'a.py,b.py'\n"
        "\n"
        "def _dispatch_modal(...):\n"
        "    files = _modal_sentinel_files(...)\n"
        "    cmd = [\n"
        "        'modal', 'run', '--detach',\n"
        "        '--require-clean-head',\n"
        "        '--lane-id', recipe.lane_id,\n"
        "        '--sentinel-files', files,\n"
        "    ]\n"
        "    return subprocess.call(cmd)\n"
    )
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=root, strict=False, verbose=False,
    )
    assert vs == []


# ─────────────────────────────────────────────────────────────────────────
# Strict mode
# ─────────────────────────────────────────────────────────────────────────


def test_strict_mode_raises_on_violation(tmp_path):
    """strict=True raises PreflightError on any violation."""
    root = _write_dispatcher(tmp_path,
        "def _dispatch_modal(...): pass\n"
    )
    with pytest.raises(PreflightError):
        check_modal_dispatch_threads_sentinel_files_per_catalog_166(
            repo_root=root, strict=True, verbose=False,
        )


def test_strict_mode_silent_on_clean_dispatcher(tmp_path):
    """strict=True is silent when contract surfaces present."""
    root = _write_dispatcher(tmp_path,
        "def _modal_sentinel_files(): return ''\n"
        "def _dispatch_modal(...):\n"
        "    cmd = ['--sentinel-files', '--require-clean-head', '--lane-id']\n"
    )
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=root, strict=True, verbose=False,
    )
    assert vs == []


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_has_zero_violations():
    """STRICT @ 0 invariant: live repo has zero violations.

    Ensures `tools/operator_authorize.py::_dispatch_modal` continues to
    thread sentinel-files, require-clean-head, and lane-id.
    """
    vs = check_modal_dispatch_threads_sentinel_files_per_catalog_166(
        repo_root=None, strict=False, verbose=False,
    )
    assert vs == [], (
        "Catalog #191 live violations re-introduced: "
        + "\n  ".join(v[:200] for v in vs[:5])
    )
