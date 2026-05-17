# SPDX-License-Identifier: MIT
"""Tests for Catalog #321:
``check_no_phantom_wyner_ziv_savings_from_research_sidecar``.

Per Q4 HALT memo 2026-05-17
(``feedback_q4_wyner_ziv_pr101_state_dict_first_empirical_anchor_build_HALTED_premise_failure_20260517.md``).

Bug class: pre-entropy probe artifacts under
``.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_*.json``
that carry positive ``deliverable_score_savings_estimate`` whose
``validation_status`` is NOT ``VALIDATED_CONTEST_MEMBER`` are phantom
scores — research-sidecar bytes are never charged by the contest rate
term ``25 * archive_bytes / 37_545_489``.

Sister of Catalog #249 (phantom-score directory) + #287 (empirical-
claim-without-evidence-tag) + #245 (Modal call_id ledger canonical
helper pattern this gate operationalizes for the prober artifact
surface).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_321_waiver_has_valid_rationale,
    check_no_phantom_wyner_ziv_savings_from_research_sidecar,
)


# ──────────────────────────────────────────────────────────────────────── #
# Helpers                                                                   #
# ──────────────────────────────────────────────────────────────────────── #


def _write_probe_artifact(
    repo_root: Path,
    name: str,
    per_substrate_results: dict[str, dict],
) -> Path:
    """Write a synthetic pre_entropy probe artifact at the canonical path."""
    out_dir = repo_root / ".omx" / "state" / "wyner_ziv_deliverability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pre_entropy_candidate_substrates_{name}.json"
    payload = {
        "schema_version": "pre_entropy_pivot_probe_v1",
        "per_substrate_results": per_substrate_results,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


# ──────────────────────────────────────────────────────────────────────── #
# Test: live-repo regression guard                                          #
# ──────────────────────────────────────────────────────────────────────── #


def test_live_repo_live_count_zero() -> None:
    """The live repo must have zero Catalog #321 violations at landing.
    The original phantom artifact was quarantined; the corrected artifact
    has all rows at deliverable=0 with validation_status set."""
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        strict=False, verbose=False
    )
    assert violations == [], f"Catalog #321 live violations: {violations}"


# ──────────────────────────────────────────────────────────────────────── #
# Test: positive flagging — phantom score from non-validated row            #
# ──────────────────────────────────────────────────────────────────────── #


def test_flags_phantom_savings_from_unvalidated_row(tmp_path: Path) -> None:
    """Synthetic artifact with deliverable > 0 + validation_status=None
    is flagged."""
    _write_probe_artifact(
        tmp_path,
        "synth_phantom",
        {
            "phantom_sub": {
                "deliverable_score_savings_estimate": 0.477,
                "validation_status": None,
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "phantom_sub" in violations[0]
    assert "0.477" in violations[0]
    assert "Catalog #321" in violations[0]


def test_flags_phantom_with_rejected_status(tmp_path: Path) -> None:
    """validation_status=REJECTED_RESEARCH_SIDECAR but deliverable > 0
    is flagged (logically inconsistent state)."""
    _write_probe_artifact(
        tmp_path,
        "synth_rejected_phantom",
        {
            "rejected_phantom": {
                "deliverable_score_savings_estimate": 1.5,
                "validation_status": "REJECTED_RESEARCH_SIDECAR",
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "REJECTED_RESEARCH_SIDECAR" in violations[0]


def test_flags_multiple_phantom_rows(tmp_path: Path) -> None:
    """One artifact with 3 phantom rows aggregates to 3 violations."""
    _write_probe_artifact(
        tmp_path,
        "synth_multi",
        {
            "phantom_a": {
                "deliverable_score_savings_estimate": 0.1,
                "validation_status": None,
            },
            "phantom_b": {
                "deliverable_score_savings_estimate": 0.2,
                "validation_status": "REJECTED_RESEARCH_SIDECAR",
            },
            "phantom_c": {
                "deliverable_score_savings_estimate": 0.3,
                "validation_status": "UNVALIDATED",
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 3


# ──────────────────────────────────────────────────────────────────────── #
# Test: negative — validated rows do NOT flag                              #
# ──────────────────────────────────────────────────────────────────────── #


def test_validated_row_with_positive_deliverable_not_flagged(tmp_path: Path) -> None:
    """validation_status=VALIDATED_CONTEST_MEMBER + deliverable > 0 → OK."""
    _write_probe_artifact(
        tmp_path,
        "synth_validated",
        {
            "valid_sub": {
                "deliverable_score_savings_estimate": 0.5,
                "validation_status": "VALIDATED_CONTEST_MEMBER",
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_zero_deliverable_row_not_flagged(tmp_path: Path) -> None:
    """deliverable=0.0 is NOT flagged regardless of validation_status."""
    _write_probe_artifact(
        tmp_path,
        "synth_zero",
        {
            "rejected_zero": {
                "deliverable_score_savings_estimate": 0.0,
                "validation_status": "REJECTED_RESEARCH_SIDECAR",
            },
            "unvalidated_zero": {
                "deliverable_score_savings_estimate": 0.0,
                "validation_status": None,
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_empty_artifact_not_flagged(tmp_path: Path) -> None:
    _write_probe_artifact(tmp_path, "empty", {})
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_no_probe_artifact_dir_not_flagged(tmp_path: Path) -> None:
    """No `.omx/state/wyner_ziv_deliverability/` dir → silent OK."""
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ──────────────────────────────────────────────────────────────────────── #
# Test: waiver semantics                                                    #
# ──────────────────────────────────────────────────────────────────────── #


def test_waiver_accepted_with_real_rationale(tmp_path: Path) -> None:
    """Waiver token in validation_reason with non-placeholder rationale is
    accepted."""
    _write_probe_artifact(
        tmp_path,
        "synth_waived",
        {
            "waived_sub": {
                "deliverable_score_savings_estimate": 0.4,
                "validation_status": None,
                "validation_reason": (
                    "# RESEARCH_SIDECAR_PHANTOM_SCORE_OK:"
                    "diagnostic-only probe for hoist-design discussion; "
                    "operator-reviewed 2026-05-17"
                ),
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_waiver_with_placeholder_rejected(tmp_path: Path) -> None:
    """Placeholder rationale `<rationale>` is rejected."""
    _write_probe_artifact(
        tmp_path,
        "synth_placeholder",
        {
            "fake_waived": {
                "deliverable_score_savings_estimate": 0.5,
                "validation_status": None,
                "validation_reason": (
                    "# RESEARCH_SIDECAR_PHANTOM_SCORE_OK:<rationale>"
                ),
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_waiver_with_reason_placeholder_rejected(tmp_path: Path) -> None:
    """Alternate placeholder `<reason>` also rejected."""
    _write_probe_artifact(
        tmp_path,
        "synth_reason_placeholder",
        {
            "fake_waived": {
                "deliverable_score_savings_estimate": 0.5,
                "validation_status": None,
                "validation_reason": (
                    "# RESEARCH_SIDECAR_PHANTOM_SCORE_OK:<reason>"
                ),
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_waiver_via_explicit_field(tmp_path: Path) -> None:
    """Waiver can live in a sister `_phantom_score_waiver` field."""
    _write_probe_artifact(
        tmp_path,
        "synth_explicit_waiver_field",
        {
            "waived_sub": {
                "deliverable_score_savings_estimate": 0.4,
                "validation_status": None,
                "_phantom_score_waiver": (
                    "# RESEARCH_SIDECAR_PHANTOM_SCORE_OK:"
                    "deliberate operator-approved diagnostic probe"
                ),
            },
        },
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ──────────────────────────────────────────────────────────────────────── #
# Test: strict-mode behavior                                                #
# ──────────────────────────────────────────────────────────────────────── #


def test_strict_mode_raises_with_catalog_321_message(tmp_path: Path) -> None:
    """Strict mode raises PreflightError mentioning Catalog #321."""
    _write_probe_artifact(
        tmp_path,
        "synth_strict_raise",
        {
            "phantom": {
                "deliverable_score_savings_estimate": 0.477,
                "validation_status": None,
            },
        },
    )
    with pytest.raises(PreflightError) as excinfo:
        check_no_phantom_wyner_ziv_savings_from_research_sidecar(
            repo_root=tmp_path, strict=True, verbose=False
        )
    msg = str(excinfo.value)
    assert "Catalog #321" in msg
    assert "phantom" in msg


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode is silent on a clean artifact set."""
    _write_probe_artifact(
        tmp_path,
        "synth_clean",
        {
            "validated_clean": {
                "deliverable_score_savings_estimate": 0.0,
                "validation_status": "VALIDATED_CONTEST_MEMBER",
            },
        },
    )
    result = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert result == []


# ──────────────────────────────────────────────────────────────────────── #
# Test: malformed JSON handled gracefully                                   #
# ──────────────────────────────────────────────────────────────────────── #


def test_malformed_json_produces_violation(tmp_path: Path) -> None:
    """Malformed JSON in a probe artifact is itself a violation."""
    out_dir = tmp_path / ".omx" / "state" / "wyner_ziv_deliverability"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pre_entropy_candidate_substrates_malformed.json").write_text(
        "{not valid json"
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "could not parse" in violations[0]


# ──────────────────────────────────────────────────────────────────────── #
# Test: sister scan — master_gradient OptimalPerPairTreatmentPlan          #
# ──────────────────────────────────────────────────────────────────────── #


def test_sister_scan_flags_wyner_ziv_treatment_citing_sidecar_path(tmp_path: Path) -> None:
    """OptimalPerPairTreatmentPlan citing a Wyner-Ziv hoist whose
    deliverability_proof_path is a .pt research sidecar is flagged."""
    consumer_dir = tmp_path / ".omx" / "state" / "master_gradient_consumers"
    consumer_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "treatments": [
            {
                "treatment_type": "wyner_ziv_hoist",
                "deliverability_proof_path": (
                    "experiments/results/pr101_codecop_sweep_20260507_codex/"
                    "pr101_decoder_state_dict.pt"
                ),
            },
        ],
    }
    (consumer_dir / "optimal_plan_20260517T000000.json").write_text(
        json.dumps(plan)
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) >= 1
    assert any("treatments" in v and ".pt" in v for v in violations)


def test_sister_scan_does_not_flag_non_wyner_ziv_treatments(tmp_path: Path) -> None:
    """Treatments that don't mention Wyner-Ziv are not flagged even if they
    cite .pt paths."""
    consumer_dir = tmp_path / ".omx" / "state" / "master_gradient_consumers"
    consumer_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "treatments": [
            {
                "treatment_type": "byte_replacement_pure",
                "deliverability_proof_path": "experiments/results/something.pt",
            },
        ],
    }
    (consumer_dir / "optimal_plan_20260517T010000.json").write_text(
        json.dumps(plan)
    )
    violations = check_no_phantom_wyner_ziv_savings_from_research_sidecar(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ──────────────────────────────────────────────────────────────────────── #
# Test: waiver helper unit tests                                            #
# ──────────────────────────────────────────────────────────────────────── #


def test_waiver_helper_accepts_real_rationale() -> None:
    txt = "# RESEARCH_SIDECAR_PHANTOM_SCORE_OK:deliberate operator-approved diagnostic"
    assert _check_321_waiver_has_valid_rationale(txt) is True


def test_waiver_helper_rejects_placeholder_rationale() -> None:
    assert _check_321_waiver_has_valid_rationale(
        "# RESEARCH_SIDECAR_PHANTOM_SCORE_OK:<rationale>"
    ) is False


def test_waiver_helper_rejects_short_rationale() -> None:
    assert _check_321_waiver_has_valid_rationale(
        "# RESEARCH_SIDECAR_PHANTOM_SCORE_OK:ok"
    ) is False


def test_waiver_helper_rejects_missing_token() -> None:
    assert _check_321_waiver_has_valid_rationale("no token here") is False
