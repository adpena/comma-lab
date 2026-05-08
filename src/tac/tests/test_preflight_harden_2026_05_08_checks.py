"""Tests for HARDEN-2026-05-08 preflight checks.

Each new check has two tests: one positive (clean tree -> 0 violations on
the live repo) and one negative (synthetic violation in a tmp tree raises
under strict=True). The live-repo positive check guards regressions; the
tmp-tree negative check guards the detector itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_137531_candidate_decoder_path_wired,
    check_admm_lagrangian_bisection_convergent,
    check_codec_pipeline_op_order_deterministic,
    check_evidence_row_has_falsification_scope_when_negative,
    check_per_tensor_K_side_info_matches_decoder_expectation,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ─── Check: ADMM Lagrangian bisection convergent ─────────────────────────


def test_admm_bisection_check_passes_on_live_repo() -> None:
    """The live tree should have 0 violations — both declared bisection
    tools have the required `for _ in range(N):` cap + tolerance break."""
    violations = check_admm_lagrangian_bisection_convergent(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_admm_bisection_strict_passes_on_live_repo() -> None:
    """Strict-mode parity check (raises if any violation)."""
    check_admm_lagrangian_bisection_convergent(strict=True, verbose=False)


# ─── Check: codec pipeline op order deterministic ────────────────────────


def test_codec_pipeline_determinism_check_passes_on_live_repo() -> None:
    violations = check_codec_pipeline_op_order_deterministic(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_codec_pipeline_determinism_strict_passes_on_live_repo() -> None:
    check_codec_pipeline_op_order_deterministic(strict=True, verbose=False)


# ─── Check: per-tensor K side-info matches decoder expectation ───────────


def test_per_tensor_K_check_passes_on_live_repo() -> None:
    violations = check_per_tensor_K_side_info_matches_decoder_expectation(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_per_tensor_K_strict_passes_on_live_repo() -> None:
    check_per_tensor_K_side_info_matches_decoder_expectation(
        strict=True, verbose=False
    )


# ─── Check: evidence row falsification scope when negative ───────────────


def test_falsification_scope_check_passes_on_live_repo() -> None:
    violations = check_evidence_row_has_falsification_scope_when_negative(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_falsification_scope_negative_detected(tmp_path: Path) -> None:
    """A row with family_falsified=False AND a negative/retired marker AND
    missing falsification_scope MUST be detected in strict mode.

    The check requires `negative_or_retired` evidence (verdict / status /
    blockers / grade containing one of: negative / retired / falsified)
    before insisting on a scope — proxy rows that don't claim negativity
    are exempt."""
    (tmp_path / "reports").mkdir(parents=True)
    bad_path = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    bad_path.write_text(
        json.dumps(
            {
                "technique": "test_bad_row",
                "family_falsified": False,
                "contest_dispatch_verdict": "negative",
                # falsification_scope missing
            }
        )
        + "\n"
    )
    with pytest.raises(PreflightError) as exc_info:
        check_evidence_row_has_falsification_scope_when_negative(
            repo_root=tmp_path, strict=True, verbose=False
        )
    assert "test_bad_row" in str(exc_info.value)
    assert "falsification_scope" in str(exc_info.value)


def test_falsification_scope_proxy_row_without_scope_passes(tmp_path: Path) -> None:
    """A proxy row with family_falsified=False but NO negative-or-retired
    marker is allowed to omit falsification_scope (no class-level claim
    being made)."""
    (tmp_path / "reports").mkdir(parents=True)
    p = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    p.write_text(
        json.dumps(
            {
                "technique": "neutral_proxy",
                "family_falsified": False,
                # no scope, no negative verdict
            }
        )
        + "\n"
    )
    violations = check_evidence_row_has_falsification_scope_when_negative(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert violations == []


def test_falsification_scope_true_value_warns(tmp_path: Path) -> None:
    """A row with family_falsified=True (without explicit grand-council
    record) is also a violation per CLAUDE.md "KILL is the LAST RESORT"."""
    (tmp_path / "reports").mkdir(parents=True)
    bad_path = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    bad_path.write_text(
        json.dumps(
            {
                "technique": "premature_kill",
                "family_falsified": True,
                "falsification_scope": "anything",
            }
        )
        + "\n"
    )
    violations = check_evidence_row_has_falsification_scope_when_negative(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("premature_kill" in v for v in violations)
    assert any("LAST RESORT" in v for v in violations)


def test_falsification_scope_clean_passes(tmp_path: Path) -> None:
    """A row with family_falsified=False AND a non-empty falsification_scope
    must pass."""
    (tmp_path / "reports").mkdir(parents=True)
    good_path = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    good_path.write_text(
        json.dumps(
            {
                "technique": "good_row",
                "family_falsified": False,
                "falsification_scope": "tested_only_config_X_at_rms_Y",
            }
        )
        + "\n"
    )
    violations = check_evidence_row_has_falsification_scope_when_negative(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert violations == []


# ─── Check: 137531 candidate decoder path wired ──────────────────────────


def test_137531_candidate_check_passes_on_live_repo() -> None:
    violations = check_137531_candidate_decoder_path_wired(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_137531_candidate_strict_passes_on_live_repo() -> None:
    check_137531_candidate_decoder_path_wired(strict=True, verbose=False)
