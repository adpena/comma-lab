# SPDX-License-Identifier: MIT
"""Tests for Catalog #238 — check_smoke_path_default_relaxes_clean_head.

DX-POLISH-WAVE 2026-05-15 (DX-4) self-protection. The gate refuses any
state of ``tools/run_modal_smoke_before_full.py`` that drops the smoke-
relaxed-clean-head contract.

Coverage:
  * Live-repo regression guard (current state must pass).
  * Positive: each required surface, when missing, surfaces a violation.
  * Negative: the canonical wired smoke wrapper passes.
  * Forbidden surface: FULL-spawn body must not export Catalog #202 env.
  * Strict mode raises PreflightDxPolishError on any violation.
  * Helper unit-tests for `_slice_function_body`.
  * Verbose output (clean + dirty).
  * Missing wrapper file surfaces a clear violation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight_dx_polish_gates import (
    DX_POLISH_GATES_CATALOG_NUMBER,
    PreflightDxPolishError,
    _slice_function_body,
    check_smoke_path_default_relaxes_clean_head,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Live-repo regression
# ---------------------------------------------------------------------------


def test_live_repo_smoke_wrapper_satisfies_catalog_238() -> None:
    """The current state of the smoke wrapper passes Catalog #238."""

    violations = check_smoke_path_default_relaxes_clean_head(
        repo_root=REPO_ROOT,
        strict=False,
        verbose=False,
    )
    assert violations == [], (
        "Live repo Catalog #238 regression: "
        + "; ".join(v[:200] for v in violations)
    )


def test_catalog_number_constant() -> None:
    assert DX_POLISH_GATES_CATALOG_NUMBER == 238


# ---------------------------------------------------------------------------
# Synthetic-corpus fixtures
# ---------------------------------------------------------------------------


_CANONICAL_SMOKE_WRAPPER = '''
"""Canonical smoke wrapper fixture for Catalog #238 tests."""


def _count_dirty_paths(repo_root):
    return 0


def _spawn_smoke_dispatch(recipe_path, env=None):
    """Smoke spawn fixture (DX-4)."""
    env = dict(env or {})
    dirty_count = _count_dirty_paths(recipe_path.parent)
    if dirty_count > 0:
        env["OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"] = "1"
        env["OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"] = "1"
        print("[smoke-before-full] DX-POLISH-WAVE Catalog #238 / DX-4: dirty")
    return env


def _spawn_full_dispatch(recipe_path):
    """Full spawn fixture (DX-4)."""
    print("[smoke-before-full] DX-POLISH-WAVE Catalog #238: full-phase strict")
    return 0
'''


def _write_wrapper(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "tools" / "run_modal_smoke_before_full.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def test_canonical_wrapper_passes(tmp_path: Path) -> None:
    _write_wrapper(tmp_path, _CANONICAL_SMOKE_WRAPPER)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert violations == []


def test_missing_wrapper_file_surfaces_violation(tmp_path: Path) -> None:
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any("missing" in v for v in violations)


# ---------------------------------------------------------------------------
# Required-surface violations (positive)
# ---------------------------------------------------------------------------


def test_missing_count_helper_surfaces_violation(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "def _count_dirty_paths", "def _NOT_THE_CANONICAL_HELPER"
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any("_count_dirty_paths" in v for v in violations)


def test_missing_intent_env_in_smoke_body_violates(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK", "OPERATOR_AUTHORIZE_OTHER_INTENT"
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any("SKIP_WHOLE_TREE_CLEAN_CHECK" in v for v in violations)


def test_missing_attestation_env_in_smoke_body_violates(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED",
        "OPERATOR_AUTHORIZE_OTHER_ATTESTATION",
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any("TRUSTED_SENTINELS_CLEAN_VERIFIED" in v for v in violations)


def test_missing_dirty_count_guard_violates(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "if dirty_count > 0:", "# guard removed"
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any(
        "positive-count guard" in v or "dirty_count" in v for v in violations
    )


def test_missing_catalog_238_token_in_smoke_body_violates(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        '"[smoke-before-full] DX-POLISH-WAVE Catalog #238 / DX-4: dirty"',
        '"[smoke-before-full] dirty tree detected"',
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any(
        "Catalog #238 reference token in smoke-spawn body" in v for v in violations
    )


def test_missing_catalog_238_token_in_full_body_violates(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        '"[smoke-before-full] DX-POLISH-WAVE Catalog #238: full-phase strict"',
        '"[smoke-before-full] full-phase strict"',
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any(
        "Catalog #238 reference token in full-spawn body" in v for v in violations
    )


def test_missing_smoke_function_violates(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "def _spawn_smoke_dispatch", "def _DELETED_SMOKE_DISPATCH"
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any("_spawn_smoke_dispatch" in v for v in violations)


def test_missing_full_function_violates(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "def _spawn_full_dispatch", "def _DELETED_FULL_DISPATCH"
    )
    _write_wrapper(tmp_path, body)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any("_spawn_full_dispatch" in v for v in violations)


# ---------------------------------------------------------------------------
# Forbidden-surface check
# ---------------------------------------------------------------------------


def test_full_phase_must_not_export_catalog_202_intent_env(tmp_path: Path) -> None:
    bad = _CANONICAL_SMOKE_WRAPPER.replace(
        '"[smoke-before-full] DX-POLISH-WAVE Catalog #238: full-phase strict"',
        (
            '"[smoke-before-full] DX-POLISH-WAVE Catalog #238: full-phase strict"\n'
            '    import os\n'
            '    os.environ["OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"] = "1"'
        ),
    )
    _write_wrapper(tmp_path, bad)
    violations = check_smoke_path_default_relaxes_clean_head(repo_root=tmp_path)
    assert any(
        "FORBIDDEN" in v and "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK" in v
        for v in violations
    )


# ---------------------------------------------------------------------------
# Strict mode + verbose output
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "def _count_dirty_paths", "def _renamed"
    )
    _write_wrapper(tmp_path, body)
    with pytest.raises(PreflightDxPolishError) as excinfo:
        check_smoke_path_default_relaxes_clean_head(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #238" in str(excinfo.value)


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    _write_wrapper(tmp_path, _CANONICAL_SMOKE_WRAPPER)
    # Should not raise.
    check_smoke_path_default_relaxes_clean_head(
        repo_root=tmp_path, strict=True
    )


def test_verbose_output_clean(tmp_path: Path, capsys) -> None:
    _write_wrapper(tmp_path, _CANONICAL_SMOKE_WRAPPER)
    check_smoke_path_default_relaxes_clean_head(
        repo_root=tmp_path, verbose=True
    )
    out = capsys.readouterr().out
    assert "[dx-polish-238] OK" in out


def test_verbose_output_dirty(tmp_path: Path, capsys) -> None:
    body = _CANONICAL_SMOKE_WRAPPER.replace(
        "def _count_dirty_paths", "def _renamed"
    )
    _write_wrapper(tmp_path, body)
    check_smoke_path_default_relaxes_clean_head(
        repo_root=tmp_path, verbose=True
    )
    out = capsys.readouterr().out
    assert "violation(s)" in out


# ---------------------------------------------------------------------------
# Helper: _slice_function_body
# ---------------------------------------------------------------------------


def test_slice_function_body_returns_only_target_function_body() -> None:
    text = (
        "def _alpha():\n"
        "    return 1\n"
        "\n"
        "def _spawn_smoke_dispatch():\n"
        "    return 'smoke'\n"
        "\n"
        "def _spawn_full_dispatch():\n"
        "    return 'full'\n"
    )
    smoke = _slice_function_body(text, "_spawn_smoke_dispatch")
    assert "smoke" in smoke
    assert "full" not in smoke
    full = _slice_function_body(text, "_spawn_full_dispatch")
    assert "full" in full


def test_slice_function_body_returns_empty_for_missing() -> None:
    assert _slice_function_body("def _other():\n    pass\n", "_missing") == ""


def test_slice_function_body_handles_eof_after_function() -> None:
    text = "def _only():\n    return 1\n"
    body = _slice_function_body(text, "_only")
    assert "return 1" in body
