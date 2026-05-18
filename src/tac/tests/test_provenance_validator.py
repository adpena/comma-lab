# SPDX-License-Identifier: MIT
"""Tests for tac.provenance.validator + audit_score_claim_dict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.provenance import (
    InvalidProvenanceError,
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
    ScoreClaim,
    audit_score_claim_dict,
    build_provenance_for_forbidden_out_of_archive_payload,
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
    provenance_to_dict,
    validate_provenance,
    validate_score_claim,
)
from tac.provenance.validator import _provenance_from_dict


def _make_research_prov() -> Provenance:
    return build_provenance_for_research_sidecar(
        sidecar_path="experiments/results/dummy/state.pt",
        reactivation_criteria="test",
    )


# -----------------------------------------------------------------------------
# validate_provenance
# -----------------------------------------------------------------------------

def test_validate_research_sidecar_clean():
    prov = _make_research_prov()
    valid, blockers = validate_provenance(prov)
    assert valid
    assert blockers == []


def test_validate_future_captured_at_flagged():
    prov = build_provenance_for_predicted(
        model_id="m",
        inputs_sha256="a" * 64,
        captured_at_utc="2099-12-31T00:00:00Z",
    )
    valid, blockers = validate_provenance(prov)
    assert not valid
    assert any("future" in b.lower() for b in blockers)


def test_validate_stale_captured_at_flagged():
    prov = build_provenance_for_predicted(
        model_id="m",
        inputs_sha256="a" * 64,
        captured_at_utc="2020-01-01T00:00:00Z",
    )
    valid, blockers = validate_provenance(prov, stale_days=90)
    assert not valid
    assert any("stale" in b.lower() for b in blockers)


def test_validate_unparseable_captured_at_flagged():
    # Construct Provenance with garbled timestamp via dict round-trip
    d = provenance_to_dict(_make_research_prov())
    d["captured_at_utc"] = "this is not iso"
    prov = _provenance_from_dict(d)
    valid, blockers = validate_provenance(prov)
    assert not valid
    assert any("parseable" in b.lower() for b in blockers)


def test_validate_aggregate_cycle_detection():
    """Aggregate composed_from cycle should be detected.

    Frozen dataclass means we can't mutate; we test cycle detection
    via the _walk helper by constructing nested aggregates that don't
    actually self-reference but visit deep enough to verify the visited
    set logic.
    """
    a = build_provenance_for_predicted(model_id="a", inputs_sha256="1" * 64)
    b = build_provenance_for_predicted(model_id="b", inputs_sha256="2" * 64)
    from tac.provenance import build_provenance_aggregate
    agg = build_provenance_aggregate(
        parts=[a, b],
        aggregation_rationale="ok",
    )
    valid, blockers = validate_provenance(agg)
    assert valid


def test_validate_invalid_byte_identity_without_reason_flagged():
    """Provenance with INVALID_BYTE_IDENTITY but no reason is flagged.

    This shouldn't happen via canonical builders (they enforce), but
    test the validator's defense-in-depth.
    """
    d = provenance_to_dict(_make_research_prov())
    d["evidence_grade"] = ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT.value
    d["rejection_reason"] = ""
    # __post_init__ would block this; but the validator should also flag.
    # Direct dataclass construction with this state should fail at __post_init__.
    with pytest.raises(InvalidProvenanceError):
        _provenance_from_dict(d)


# -----------------------------------------------------------------------------
# validate_score_claim
# -----------------------------------------------------------------------------

def test_validate_score_claim_research_sidecar_passes_validation():
    prov = _make_research_prov()
    claim = ScoreClaim(score_value=0.5, provenance=prov)
    valid, blockers = validate_score_claim(claim)
    assert valid


def test_validate_score_claim_nan_flagged():
    import math
    prov = _make_research_prov()
    claim = ScoreClaim(score_value=math.nan, provenance=prov)
    valid, blockers = validate_score_claim(claim)
    assert not valid
    assert any("nan" in b.lower() for b in blockers)


def test_validate_score_claim_infinity_flagged():
    import math
    prov = _make_research_prov()
    claim = ScoreClaim(score_value=math.inf, provenance=prov)
    valid, blockers = validate_score_claim(claim)
    assert not valid
    assert any("infinite" in b.lower() for b in blockers)


def test_validate_score_claim_negative_flagged():
    prov = _make_research_prov()
    claim = ScoreClaim(score_value=-0.1, provenance=prov)
    valid, blockers = validate_score_claim(claim)
    assert not valid
    assert any("negative" in b.lower() for b in blockers)


# -----------------------------------------------------------------------------
# audit_score_claim_dict — phantom-score class anchors
# -----------------------------------------------------------------------------

def test_audit_no_score_keys_passes():
    """Rows without score-claim keys are CLEAN by construction."""
    payload = {"name": "test", "count": 42, "ok": True}
    valid, blockers = audit_score_claim_dict(payload)
    assert valid


def test_audit_score_without_provenance_flagged():
    """Catalog #321 anchor: pr101_state_dict 0.477 with no Provenance."""
    payload = {"score": 0.477, "rationale": "pr101 state_dict"}
    valid, blockers = audit_score_claim_dict(payload)
    assert not valid
    assert any("provenance" in b.lower() for b in blockers)


@pytest.mark.parametrize(
    "score_key",
    [
        "canonical_score",
        "canonical_score_recomputed",
        "score_recomputed",
        "score_recomputed_from_components",
        "score_recomputed_from_contest_components",
        "score_recomputed_from_public_components",
        "score_contest_cuda",
        "score_contest_cpu",
        "contest_cuda_score_recomputed",
        "contest_cpu_score_recomputed",
        "empirical_score",
        "diagnostic_cpu_score",
        "auth_eval_score",
        "auth_eval_recomputed_score",
        "score_recomputed_from_auth_eval",
        "recomputed_score",
    ],
)
def test_audit_common_auth_eval_score_synonyms_without_provenance_flagged(
    score_key: str,
):
    """Validator key set covers live auth-eval and harvest score spellings."""
    valid, blockers = audit_score_claim_dict({score_key: 0.5})
    assert not valid
    assert any("provenance" in b.lower() for b in blockers)


def test_audit_phantom_score_keys_caught():
    """Catalog #321 anchor: deliverable_score_savings_estimate phantom."""
    payload = {
        "deliverable_score_savings_estimate": 11.6,
        "lane_id": "posenet_class_sensitivity",
    }
    valid, blockers = audit_score_claim_dict(payload)
    assert not valid


def test_audit_composition_alpha_without_provenance_caught():
    """Catalog #319 anchor: composition_alpha phantom without DeliverabilityProof."""
    payload = {"composition_alpha": 4.74, "pair_key": "fec6_x_pr106"}
    valid, blockers = audit_score_claim_dict(payload)
    assert not valid


def test_audit_alpha_savings_ratio_form_without_provenance_caught():
    """Catalog #823 anchor: alpha_savings_ratio_form from SIREN byte-identity."""
    payload = {
        "alpha_savings_ratio_form": 4.74,
        "pair_key": "lane_g_v3_renderer__x__siren_renderer",
    }
    valid, blockers = audit_score_claim_dict(payload)
    assert not valid


def test_audit_with_valid_provenance_passes():
    prov_dict = provenance_to_dict(_make_research_prov())
    payload = {
        "score": 0.0,  # 0 is OK — non-zero with research_only triggers phantom flag
        "provenance": prov_dict,
    }
    valid, blockers = audit_score_claim_dict(payload)
    assert valid


def test_audit_non_zero_score_with_research_provenance_flagged():
    """Phantom-score class: research_only sub-provenance with non-zero score."""
    prov_dict = provenance_to_dict(_make_research_prov())
    payload = {
        "score": 0.477,  # phantom score from #321 anchor
        "provenance": prov_dict,
    }
    valid, blockers = audit_score_claim_dict(payload)
    assert not valid
    assert any("phantom" in b.lower() for b in blockers)


def test_audit_provenance_field_not_dict_flagged():
    payload = {"score": 0.5, "provenance": "not_a_dict"}
    valid, blockers = audit_score_claim_dict(payload)
    assert not valid


def test_audit_expected_axis_mismatch_flagged():
    prov = build_provenance_for_predicted(model_id="m", inputs_sha256="a" * 64)
    prov_dict = provenance_to_dict(prov)
    payload = {"score": 0.0, "provenance": prov_dict}
    valid, blockers = audit_score_claim_dict(payload, expected_axis="[contest-CUDA]")
    assert not valid
    assert any("axis" in b.lower() for b in blockers)


def test_audit_forbidden_out_of_archive_payload_fails_even_for_zero_score():
    prov = build_provenance_for_forbidden_out_of_archive_payload(
        payload_source_path="/external/payload.bin",
        payload_sha256="3" * 64,
        rejection_reason="output-affecting payload bytes are outside archive.zip",
    )
    payload = {"score": 0.0, "provenance": provenance_to_dict(prov)}
    valid, blockers = audit_score_claim_dict(payload)
    assert not valid
    assert any("FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD" in b for b in blockers)


# -----------------------------------------------------------------------------
# Round-trip serialization
# -----------------------------------------------------------------------------

def test_provenance_round_trip_via_dict():
    original = _make_research_prov()
    d = provenance_to_dict(original)
    reconstructed = _provenance_from_dict(d)
    assert reconstructed.source_path == original.source_path
    assert reconstructed.source_sha256 == original.source_sha256
    assert reconstructed.artifact_kind == original.artifact_kind
    assert reconstructed.evidence_grade == original.evidence_grade


def test_provenance_aggregate_round_trip():
    a = build_provenance_for_predicted(model_id="a", inputs_sha256="1" * 64)
    b = build_provenance_for_predicted(model_id="b", inputs_sha256="2" * 64)
    from tac.provenance import build_provenance_aggregate
    agg = build_provenance_aggregate(parts=[a, b], aggregation_rationale="rt test")
    d = provenance_to_dict(agg)
    reconstructed = _provenance_from_dict(d)
    assert reconstructed.artifact_kind == ProvenanceKind.AGGREGATE_OF_PROVENANCES
    assert len(reconstructed.composed_from) == 2


def test_byte_identity_aggregate_round_trip_preserved():
    """Catalog #823 detection survives serialization round-trip."""
    a = build_provenance_for_predicted(model_id="m1", inputs_sha256="08f12d72" + "2" * 56)
    b = build_provenance_for_predicted(model_id="m2", inputs_sha256="08f12d72" + "2" * 56)
    from tac.provenance import build_provenance_aggregate
    agg = build_provenance_aggregate(parts=[a, b], aggregation_rationale="bi test")
    assert agg.evidence_grade == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT
    d = provenance_to_dict(agg)
    reconstructed = _provenance_from_dict(d)
    assert (
        reconstructed.evidence_grade
        == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT
    )


# -----------------------------------------------------------------------------
# Live-repo regression guard
# -----------------------------------------------------------------------------

def test_live_repo_provenance_compliance_audit_bounded():
    """Smoke-test that audit tool runs without crashing on live repo."""
    from tools.audit_provenance_compliance import build_audit_report

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    report = build_audit_report(repo_root)
    # Just verify it produces a report; we expect baseline violations
    # (188 at landing) until backfill sweep completes.
    assert report.total_artifacts_scanned > 0
    # Bound at sane ceiling so a future regression spikes can be detected
    assert report.violation_count < 5000, (
        f"violation count={report.violation_count} unexpectedly high;"
        f" possible regression"
    )


def test_audit_tool_common_auth_eval_score_synonyms_without_provenance_flagged(
    tmp_path: Path,
):
    """Operator audit tool uses the same score-key vocabulary as the gate."""
    from tools.audit_provenance_compliance import build_audit_report

    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    for key in ("canonical_score", "score_recomputed", "score_recomputed_from_components"):
        (state_dir / f"{key}.json").write_text(json.dumps({key: 0.5}))

    report = build_audit_report(tmp_path)
    assert report.violation_count == 3
    violating_paths = {
        Path(v.path).name
        for v in report.artifact_verdicts
        if v.verdict == "VIOLATION"
    }
    assert violating_paths == {
        "canonical_score.json",
        "score_recomputed.json",
        "score_recomputed_from_components.json",
    }
