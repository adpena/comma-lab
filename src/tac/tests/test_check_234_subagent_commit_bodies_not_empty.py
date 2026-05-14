# SPDX-License-Identifier: MIT
"""Catalog #234 integration tests.

The behavioral fixtures for Catalog #234 live in
``test_check_234_235_meta_commit_gates.py``.  This file protects the wiring
bug that appeared during the R1 landing: the gate was registered through the
parallel runner and then called again inline, doubling preflight work.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from tac.preflight import (
    _CHECK_234_BODY_MARKER_TOKENS,
    _CHECK_234_DISCIPLINE_CUTOFF_UTC,
    PreflightError,
    check_subagent_commit_bodies_not_empty,
)


def test_check_234_canonical_body_markers_present() -> None:
    """The canonical body markers tuple must contain key tokens."""
    expected = {
        "Co-Authored-By",
        "subagent_checkpoint.py",
        "subagent_progress.jsonl",
        "checkpoint discipline",
        "CHECKPOINT_DISCIPLINE_WAIVED",
        "Lane:",
        "Memory:",
        "feedback_",
        "EMPTY_BODY_OK",
    }
    for token in expected:
        assert token in _CHECK_234_BODY_MARKER_TOKENS, (
            f"Catalog #234 must accept token {token!r}"
        )


def test_check_234_cutoff_is_utc_iso8601() -> None:
    """The cutoff must be a parseable UTC ISO8601 string."""
    assert _CHECK_234_DISCIPLINE_CUTOFF_UTC.endswith("Z")
    assert "T" in _CHECK_234_DISCIPLINE_CUTOFF_UTC
    # Format: YYYY-MM-DDTHH:MM:SSZ
    assert len(_CHECK_234_DISCIPLINE_CUTOFF_UTC) == 20


def test_check_234_empty_repo_has_no_violations(tmp_path) -> None:
    """No serializer log means no scoped subagent commits."""
    result = check_subagent_commit_bodies_not_empty(
        strict=False,
        verbose=False,
        repo_root=tmp_path,
    )
    assert result == []


def test_check_234_strict_empty_repo_is_clean(tmp_path) -> None:
    """Strict mode should not raise when no scoped commits exist."""
    try:
        result = check_subagent_commit_bodies_not_empty(
            strict=True,
            verbose=False,
            repo_root=tmp_path,
        )
    except PreflightError as exc:  # pragma: no cover - clearer failure
        pytest.fail(f"Catalog #234 raised unexpectedly: {exc}")
    assert result == []


def test_check_234_preflight_all_registers_gate_once() -> None:
    """Catalog #234 should be registered once, through the parallel runner."""
    from tac import preflight as pf

    src = inspect.getsource(pf.preflight_all)
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check_subagent_commit_bodies_not_empty"
    ]
    assert len(calls) == 1


def test_check_235_preflight_all_registers_gate_once() -> None:
    """Catalog #235 should not run once in parallel and again inline."""
    from tac import preflight as pf

    src = inspect.getsource(pf.preflight_all)
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check_no_sha_prefix_length_mismatch_comparisons"
    ]
    assert len(calls) == 1


def test_check_234_preflight_all_warn_only_callsite() -> None:
    """Catalog #234 remains warn-only until the strict-flip sweep lands."""
    from tac import preflight as pf

    src = inspect.getsource(pf.preflight_all)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "check_subagent_commit_bodies_not_empty"
        ):
            strict_kw = next(
                (kw for kw in node.keywords if kw.arg == "strict"),
                None,
            )
            assert isinstance(strict_kw, ast.keyword)
            assert isinstance(strict_kw.value, ast.Constant)
            assert strict_kw.value.value is False
            return
    pytest.fail("Catalog #234 is not wired into preflight_all()")
