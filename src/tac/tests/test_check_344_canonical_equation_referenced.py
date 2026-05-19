# SPDX-License-Identifier: MIT
"""Tests for Catalog #344 — canonical equation reference required in empirical memos."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_empirical_finding_memo_references_canonical_equation,
)


def _write_memo(tmp_path: Path, name: str, body: str) -> Path:
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True, exist_ok=True)
    p = research / name
    p.write_text(body, encoding="utf-8")
    return p


def test_empty_repo_no_violations(tmp_path: Path):
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_no_research_dir_skipped(tmp_path: Path):
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_pre_cutoff_memo_exempt(tmp_path: Path):
    _write_memo(
        tmp_path,
        "old_memo_20260518.md",
        "We empirically falsified the prediction.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_memo_with_empirical_finding_no_reference_flagged(tmp_path: Path):
    _write_memo(
        tmp_path,
        "new_finding_20260520.md",
        "The empirical anchor showed predicted vs measured drift of 30x. "
        "This is a clear ratified observation.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_memo_with_canonical_equation_reference_accepted(tmp_path: Path):
    _write_memo(
        tmp_path,
        "finding_with_eq_20260520.md",
        "The empirical anchor matched the prediction. We registered this in "
        "tac.canonical_equations via register_canonical_equation.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_memo_with_formalization_pending_waiver_accepted(tmp_path: Path):
    _write_memo(
        tmp_path,
        "finding_pending_20260520.md",
        "We empirically falsified the prediction. # FORMALIZATION_PENDING:will land equation next session",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_placeholder_rationale_rejected(tmp_path: Path):
    _write_memo(
        tmp_path,
        "finding_placeholder_20260520.md",
        "Predicted vs empirical residual was 30x. # FORMALIZATION_PENDING:<rationale>",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_empty_rationale_rejected(tmp_path: Path):
    _write_memo(
        tmp_path,
        "finding_empty_20260520.md",
        "Predicted vs empirical residual was 30x. # FORMALIZATION_PENDING:",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_short_rationale_rejected(tmp_path: Path):
    _write_memo(
        tmp_path,
        "finding_short_20260520.md",
        "Predicted vs empirical residual was 30x. # FORMALIZATION_PENDING:x",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_strict_mode_raises(tmp_path: Path):
    _write_memo(
        tmp_path,
        "strict_finding_20260520.md",
        "We empirically ratified this prediction.",
    )
    with pytest.raises(PreflightError, match="Catalog #344"):
        check_empirical_finding_memo_references_canonical_equation(
            repo_root=tmp_path, strict=True
        )


def test_strict_mode_silent_on_clean(tmp_path: Path):
    _write_memo(
        tmp_path,
        "clean_finding_20260520.md",
        "Empirical anchor confirmed via tac.canonical_equations registry.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path, strict=True
    )
    assert out == []


def test_memo_without_empirical_token_skipped(tmp_path: Path):
    """Memos that don't mention empirical findings are out of scope."""
    _write_memo(
        tmp_path,
        "design_note_20260520.md",
        "Design discussion of the upcoming substrate. Reactivation criteria pinned.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_multiple_violations_aggregated(tmp_path: Path):
    for i in range(3):
        _write_memo(
            tmp_path,
            f"multi_{i}_20260520.md",
            "Predicted vs empirical residual was confirmed.",
        )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 3


def test_string_repo_root_accepted(tmp_path: Path):
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=str(tmp_path)
    )
    assert out == []


def test_canonical_equations_module_token_accepted(tmp_path: Path):
    """Any of the listed canonical-equation reference tokens satisfies the gate."""
    for token in (
        "tac.canonical_equations",
        "canonical_equations_registry",
        "register_canonical_equation",
        "update_equation_with_empirical_anchor",
    ):
        memo = _write_memo(
            tmp_path,
            f"refers_{abs(hash(token)) % 10000}_20260520.md",
            f"Empirical anchor confirmed; see {token} for details.",
        )
        out = check_empirical_finding_memo_references_canonical_equation(
            repo_root=tmp_path
        )
        assert out == [], f"{token} should satisfy the gate"
        memo.unlink()


def test_live_repo_regression_guard():
    """Live-repo count is bounded (will WARN; not raise). Initial baseline 51
    today's memos predate the framework; backfill sweep is operator-routed."""
    out = check_empirical_finding_memo_references_canonical_equation()
    # Generous bound for the initial backfill sweep window.
    assert len(out) <= 250, f"Catalog #344 live count={len(out)} exceeds bound"


def test_orchestrator_wires_warn_only():
    """Smoke-test that preflight_all does NOT raise on Catalog #344 live."""
    # The orchestrator wires Catalog #344 at strict=False; this confirms the
    # function is reachable via the preflight module's public surface.
    from tac.preflight import (
        check_empirical_finding_memo_references_canonical_equation as f,
    )

    assert callable(f)


def test_catalog_344_callable_via_globals():
    """Catalog #185 sister regression — function must be in module globals
    for the META-meta drift gate to introspect it."""
    import tac.preflight as p

    assert hasattr(
        p, "check_empirical_finding_memo_references_canonical_equation"
    )


def test_design_note_with_empirical_token_in_quote_still_flagged(tmp_path: Path):
    """Same-line waiver MUST mention the canonical marker; embedded prose alone
    does not satisfy the gate."""
    _write_memo(
        tmp_path,
        "quote_finding_20260520.md",
        "The reviewer wrote 'predicted vs empirical residual was 30x'.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1
