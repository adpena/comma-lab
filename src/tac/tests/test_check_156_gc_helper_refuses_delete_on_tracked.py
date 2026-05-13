"""Tests for Catalog #156 — check_gc_helper_refuses_delete_on_tracked_paths.

2026-05-12 (subagent F, Wave 1) — sister gate of Catalog #154. Where #154
refuses bulk-delete under ``experiments/results/`` outside the canonical
helper, #156 refuses the canonical helper ITSELF from being used in a way
that strips the Part-2 ``TrackedDeleteRefusedError`` defense-in-depth.

Memory: feedback_gc_fix_and_commit_swap_class_protect_landed_20260512.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_gc_helper_refuses_delete_on_tracked_paths,
)
from tac.source_index import source_index_context


def test_check_156_live_count_zero():
    """The check MUST have 0 live violations at landing (strict-flip atom)."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert violations == [], f"Live violations should be 0, got: {violations}"


def test_check_156_detects_external_execute_plan_call_without_handler(tmp_path):
    """A new tool that imports `execute_plan` and calls it without
    catching ``TrackedDeleteRefusedError`` is a violation."""
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_consumer.py"
    bad.write_text(
        "from tools.gc_experiments_results import execute_plan\n"
        "plan = {'would_delete': []}\n"
        "execute_plan(plan, repo_root='.', operator_approved='x:y')\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("bad_consumer.py" in v for v in violations)
    with pytest.raises(PreflightError):
        check_gc_helper_refuses_delete_on_tracked_paths(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_156_detects_external_execute_plan_call_with_source_index(tmp_path):
    """The indexed full-preflight path must preserve the strict semantics."""
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_indexed_consumer.py"
    bad.write_text(
        "from tools.gc_experiments_results import execute_plan\n"
        "execute_plan({'would_delete': []}, repo_root='.', operator_approved='x:y')\n"
    )
    with source_index_context(tmp_path):
        violations = check_gc_helper_refuses_delete_on_tracked_paths(
            repo_root=tmp_path, strict=False, verbose=False
        )
    assert any("bad_indexed_consumer.py" in v for v in violations)


def test_check_156_accepts_exception_handler(tmp_path):
    """A caller that wraps ``execute_plan`` in a try/except for
    ``TrackedDeleteRefusedError`` is acceptable."""
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "ok_consumer.py"
    ok.write_text(
        "from tools.gc_experiments_results import (\n"
        "    execute_plan, TrackedDeleteRefusedError\n"
        ")\n"
        "try:\n"
        "    execute_plan({'would_delete': []}, repo_root='.', operator_approved='x:y')\n"
        "except TrackedDeleteRefusedError as exc:\n"
        "    print('refused:', exc)\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_156_accepts_same_line_waiver(tmp_path):
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "waiver_ok.py"
    ok.write_text(
        "from tools.gc_experiments_results import execute_plan\n"
        "execute_plan({'would_delete': []}, repo_root='.', operator_approved='x:y')  # GC_TRACKED_DELETE_OK:operator-reviewed\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_156_skips_canonical_helper_and_preflight(tmp_path):
    """The canonical helper file itself is exempt; src/tac/preflight.py too."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "src" / "tac").mkdir(parents=True)
    canonical = tmp_path / "tools" / "gc_experiments_results.py"
    canonical.write_text(
        "def execute_plan(plan, *, repo_root, operator_approved, verbose=True): ...\n"
    )
    preflight = tmp_path / "src" / "tac" / "preflight.py"
    preflight.write_text(
        "# this file references execute_plan in its check signatures\n"
        "# execute_plan( ... ) — should not flag\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_156_skips_test_files(tmp_path):
    """Test files for the canonical helper are exempt — they invoke
    execute_plan on tmp_path fixtures."""
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True)
    t = tmp_path / "src" / "tac" / "tests" / "test_gc_experiments_results.py"
    t.write_text(
        "from tools.gc_experiments_results import execute_plan\n"
        "execute_plan({'would_delete': []}, repo_root='.', operator_approved='x:y')\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_156_skips_intake_clones(tmp_path):
    """Vendored public-PR intake clones are out of scope."""
    intake = tmp_path / "experiments" / "results" / "public_pr95_intake_codex"
    intake.mkdir(parents=True)
    bad = intake / "vendored_call.py"
    bad.write_text(
        "from tools.gc_experiments_results import execute_plan\n"
        "execute_plan({}, repo_root='.', operator_approved='x:y')\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_156_detects_classify_results_dirs_call(tmp_path):
    """Other public API names should also trigger: classify_results_dirs."""
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_classifier.py"
    bad.write_text(
        "from tools.gc_experiments_results import classify_results_dirs\n"
        "rows = classify_results_dirs(root='.')\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("bad_classifier.py" in v for v in violations)


def test_check_156_detects_build_gc_plan_call(tmp_path):
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_planner.py"
    bad.write_text(
        "from tools.gc_experiments_results import build_gc_plan\n"
        "p = build_gc_plan([])\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("bad_planner.py" in v for v in violations)


def test_check_156_unrelated_callsites_pass(tmp_path):
    """A file that mentions `execute_plan` in a string but doesn't call
    the helper API should not be flagged."""
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "noise.py"
    ok.write_text(
        "# this file mentions execute_plan in a docstring but does not call it\n"
        "msg = 'see execute_plan documentation'\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # The textual filter matches "execute_plan" in a string; but the
    # callsite detection requires "execute_plan(" — so it won't flag.
    assert violations == []


def test_check_156_strict_raises_on_any_violation(tmp_path):
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_single.py"
    bad.write_text(
        "from tools.gc_experiments_results import execute_plan\n"
        "execute_plan({}, repo_root='.', operator_approved='x:y')\n"
    )
    with pytest.raises(PreflightError):
        check_gc_helper_refuses_delete_on_tracked_paths(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_156_acceptance_classify_in_handler(tmp_path):
    """A consumer of classify_results_dirs that catches TrackedDeleteRefusedError
    somewhere in the file is acceptable (file-level exception proxy)."""
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "ok_classify_in_handler.py"
    ok.write_text(
        "from tools.gc_experiments_results import classify_results_dirs, TrackedDeleteRefusedError\n"
        "try:\n"
        "    rows = classify_results_dirs(root='.')\n"
        "except TrackedDeleteRefusedError:\n"
        "    rows = []\n"
    )
    violations = check_gc_helper_refuses_delete_on_tracked_paths(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []
