# SPDX-License-Identifier: MIT
"""Tests for Recursive Self-Reflection Protocol canonical helpers (Catalog #363).

Covers Surface 2 of the protocol landing per
``.omx/research/council_recursive_self_reflection_protocol_design_20260526T133600Z.md``:

* :class:`EmpiricalVerificationStatus` 4-value taxonomy
* :class:`AssumptionEmpiricalVerification` frozen dataclass + invariants
* :func:`classify_assumption_verification_status_from_evidence` precedence
* :func:`extract_unverified_assumptions` round 2 helper
* :func:`verdict_status_requires_provisional_marker` round 3 predicate
* :func:`query_self_reflection_history_for_deliberation` operator audit helper

Mirrors the test pattern of
:mod:`src.tac.tests.test_council_continual_learning` (Catalog #300 sister).

Verified-against: canonical design memo
council_recursive_self_reflection_protocol_design_20260526T133600Z.md §2.1-§2.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.council_continual_learning import (
    AssumptionEmpiricalVerification,
    AssumptionVerificationValidationError,
    CouncilDeliberationRecord,
    CouncilTier,
    EmpiricalVerificationStatus,
    MAX_SELF_REFLECTION_ROUNDS,
    UNVERIFIED_VERIFICATION_STATUSES,
    VALID_EMPIRICAL_VERIFICATION_STATUSES,
    append_council_anchor,
    classify_assumption_verification_status_from_evidence,
    extract_unverified_assumptions,
    query_self_reflection_history_for_deliberation,
    verdict_status_requires_provisional_marker,
)


# ──────────────────────────────────────────────────────────────────────
# Canonical 4-value taxonomy invariants
# ──────────────────────────────────────────────────────────────────────


def test_taxonomy_has_exactly_four_values():
    """Canonical taxonomy MUST have exactly 4 values per design memo §2.2."""
    assert len(VALID_EMPIRICAL_VERIFICATION_STATUSES) == 4


def test_taxonomy_includes_all_four_canonical_constants():
    """4 canonical sentinel constants MUST all be in VALID set."""
    assert EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION in VALID_EMPIRICAL_VERIFICATION_STATUSES
    assert EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR in VALID_EMPIRICAL_VERIFICATION_STATUSES
    assert EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE in VALID_EMPIRICAL_VERIFICATION_STATUSES
    assert EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION in VALID_EMPIRICAL_VERIFICATION_STATUSES


def test_unverified_set_has_exactly_two_values():
    """The 2 unverified statuses gate the verdict per design memo §2.2."""
    assert len(UNVERIFIED_VERIFICATION_STATUSES) == 2
    assert EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE in UNVERIFIED_VERIFICATION_STATUSES
    assert EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION in UNVERIFIED_VERIFICATION_STATUSES


def test_verified_states_not_in_unverified_set():
    """The 2 VERIFIED states must NOT be in the gate-triggering set."""
    assert EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION not in UNVERIFIED_VERIFICATION_STATUSES
    assert EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR not in UNVERIFIED_VERIFICATION_STATUSES


def test_max_self_reflection_rounds_is_five():
    """R12-D lens-coverage Zipf-decay cycle-bound per design memo §2.3."""
    assert MAX_SELF_REFLECTION_ROUNDS == 5


# ──────────────────────────────────────────────────────────────────────
# AssumptionEmpiricalVerification dataclass invariants
# ──────────────────────────────────────────────────────────────────────


def test_dataclass_constructs_with_full_canonical_record():
    """Round-trip with all 5 fields including evidence_artifact."""
    ae = AssumptionEmpiricalVerification(
        assumption="Z6 uses MLX AdamW",
        classification="HARD-EARNED",
        empirical_verification_status=EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
        rationale="Source inspection confirms optimizer = mlx.optimizers.AdamW(...)",
        evidence_artifact="long_training_canonical.py:147",
    )
    assert ae.assumption == "Z6 uses MLX AdamW"
    assert ae.evidence_artifact == "long_training_canonical.py:147"


def test_dataclass_rejects_empty_assumption():
    with pytest.raises(AssumptionVerificationValidationError, match="assumption"):
        AssumptionEmpiricalVerification(
            assumption="",
            classification="HARD-EARNED",
            empirical_verification_status=EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            rationale="x",
            evidence_artifact="foo.py:1",
        )


def test_dataclass_rejects_empty_classification():
    with pytest.raises(AssumptionVerificationValidationError, match="classification"):
        AssumptionEmpiricalVerification(
            assumption="X",
            classification="",
            empirical_verification_status=EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            rationale="x",
            evidence_artifact="foo.py:1",
        )


def test_dataclass_rejects_invalid_verification_status():
    with pytest.raises(AssumptionVerificationValidationError, match="empirical_verification_status"):
        AssumptionEmpiricalVerification(
            assumption="X",
            classification="HARD-EARNED",
            empirical_verification_status="NOT_A_VALID_STATUS",
            rationale="x",
            evidence_artifact="foo.py:1",
        )


def test_dataclass_rejects_empty_rationale():
    with pytest.raises(AssumptionVerificationValidationError, match="rationale"):
        AssumptionEmpiricalVerification(
            assumption="X",
            classification="HARD-EARNED",
            empirical_verification_status=EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            rationale="",
            evidence_artifact="foo.py:1",
        )


def test_verified_via_source_inspection_requires_evidence_artifact():
    """VERIFIED_VIA_SOURCE_INSPECTION + None evidence_artifact = invariant violation."""
    with pytest.raises(AssumptionVerificationValidationError, match="evidence_artifact"):
        AssumptionEmpiricalVerification(
            assumption="X",
            classification="HARD-EARNED",
            empirical_verification_status=EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            rationale="X",
            evidence_artifact=None,
        )


def test_verified_via_empirical_anchor_requires_evidence_artifact():
    """VERIFIED_VIA_EMPIRICAL_ANCHOR + None evidence_artifact = invariant violation."""
    with pytest.raises(AssumptionVerificationValidationError, match="evidence_artifact"):
        AssumptionEmpiricalVerification(
            assumption="X",
            classification="HARD-EARNED",
            empirical_verification_status=EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR,
            rationale="X",
            evidence_artifact=None,
        )


def test_inferred_from_literature_allows_no_evidence_artifact():
    """INFERRED status DOES NOT require evidence_artifact (it's optional)."""
    ae = AssumptionEmpiricalVerification(
        assumption="X",
        classification="HARD-EARNED",
        empirical_verification_status=EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE,
        rationale="Higham 2002",
        evidence_artifact=None,
    )
    assert ae.evidence_artifact is None


def test_assumed_awaiting_verification_allows_no_evidence_artifact():
    """ASSUMED status DOES NOT require evidence_artifact (it's optional)."""
    ae = AssumptionEmpiricalVerification(
        assumption="X",
        classification="CARGO-CULTED",
        empirical_verification_status=EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
        rationale="No source citation; operating-within unverified",
        evidence_artifact=None,
    )
    assert ae.evidence_artifact is None


def test_as_dict_round_trip():
    """Dict serialization preserves all fields."""
    ae = AssumptionEmpiricalVerification(
        assumption="X",
        classification="HARD-EARNED",
        empirical_verification_status=EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
        rationale="Y",
        evidence_artifact="foo.py:1",
    )
    d = ae.as_dict()
    ae2 = AssumptionEmpiricalVerification.from_dict(d)
    assert ae2 == ae


def test_as_dict_omits_none_evidence_artifact():
    """None evidence_artifact omitted from dict for back-compat with legacy schema."""
    ae = AssumptionEmpiricalVerification(
        assumption="X",
        classification="CARGO-CULTED",
        empirical_verification_status=EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
        rationale="Y",
        evidence_artifact=None,
    )
    d = ae.as_dict()
    assert "evidence_artifact" not in d


def test_from_dict_legacy_row_backward_compat():
    """Legacy rows lacking empirical_verification_status auto-classify INFERRED."""
    legacy = {
        "assumption": "X",
        "classification": "CARGO-CULTED",
        "rationale": "Y",
    }
    ae = AssumptionEmpiricalVerification.from_dict(legacy)
    assert ae.empirical_verification_status == EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE


def test_from_dict_legacy_row_with_empty_string_status_backfills_inferred():
    """Empty string status also auto-classifies INFERRED."""
    legacy = {
        "assumption": "X",
        "classification": "HARD-EARNED",
        "rationale": "Y",
        "empirical_verification_status": "",
    }
    ae = AssumptionEmpiricalVerification.from_dict(legacy)
    assert ae.empirical_verification_status == EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE


# ──────────────────────────────────────────────────────────────────────
# classify_assumption_verification_status_from_evidence precedence
# ──────────────────────────────────────────────────────────────────────


def test_classifier_returns_source_inspection_when_source_present():
    """Source artifact has highest precedence."""
    status = classify_assumption_verification_status_from_evidence(
        source_artifact="long_training_canonical.py:147",
    )
    assert status == EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION


def test_classifier_returns_empirical_anchor_when_only_anchor():
    """Empirical anchor when no source artifact."""
    status = classify_assumption_verification_status_from_evidence(
        empirical_anchor="commit 05c07aa40",
    )
    assert status == EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR


def test_classifier_returns_inferred_when_only_literature():
    """Literature citation when no source/anchor."""
    status = classify_assumption_verification_status_from_evidence(
        literature_citation="Higham 2002",
    )
    assert status == EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE


def test_classifier_returns_assumed_when_no_evidence():
    """No evidence at all = ASSUMED_AWAITING_VERIFICATION."""
    status = classify_assumption_verification_status_from_evidence()
    assert status == EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION


def test_classifier_precedence_source_beats_anchor():
    """Source artifact beats empirical anchor when both present."""
    status = classify_assumption_verification_status_from_evidence(
        source_artifact="foo.py:1",
        empirical_anchor="commit abc",
    )
    assert status == EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION


def test_classifier_precedence_anchor_beats_literature():
    """Empirical anchor beats literature citation when both present."""
    status = classify_assumption_verification_status_from_evidence(
        empirical_anchor="commit abc",
        literature_citation="Higham 2002",
    )
    assert status == EmpiricalVerificationStatus.VERIFIED_VIA_EMPIRICAL_ANCHOR


def test_classifier_empty_string_evidence_treated_as_missing():
    """Empty strings count as missing evidence."""
    status = classify_assumption_verification_status_from_evidence(
        source_artifact="",
        empirical_anchor="",
        literature_citation="",
    )
    assert status == EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION


def test_classifier_whitespace_only_evidence_treated_as_missing():
    """Whitespace-only evidence counts as missing."""
    status = classify_assumption_verification_status_from_evidence(
        source_artifact="   ",
    )
    assert status == EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION


# ──────────────────────────────────────────────────────────────────────
# extract_unverified_assumptions Round 2 helper
# ──────────────────────────────────────────────────────────────────────


def _make_record(
    *,
    tier=CouncilTier.T2,
    adv_verdict_dicts=(),
) -> CouncilDeliberationRecord:
    """Build a CouncilDeliberationRecord fixture with default required fields."""
    return CouncilDeliberationRecord(
        deliberation_id="test_001",
        topic="test",
        council_tier=tier,
        council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
        council_quorum_met=True,
        council_verdict="PROCEED",
        council_assumption_adversary_verdict=adv_verdict_dicts,
        predicted_mission_contribution=(
            "apparatus_maintenance" if tier != CouncilTier.T1 else None
        ),
    )


def test_extract_unverified_returns_empty_when_all_verified():
    """All-VERIFIED record has no unverified assumptions."""
    rec = _make_record(adv_verdict_dicts=(
        {
            "assumption": "X",
            "classification": "HARD-EARNED",
            "rationale": "Z",
            "empirical_verification_status": EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            "evidence_artifact": "foo.py:1",
        },
    ))
    assert extract_unverified_assumptions(rec) == []


def test_extract_unverified_catches_inferred():
    """INFERRED_FROM_DOMAIN_LITERATURE is unverified per design memo §2.2."""
    rec = _make_record(adv_verdict_dicts=(
        {
            "assumption": "X",
            "classification": "HARD-EARNED",
            "rationale": "Higham 2002",
            "empirical_verification_status": EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE,
        },
    ))
    unverified = extract_unverified_assumptions(rec)
    assert len(unverified) == 1
    assert unverified[0].assumption == "X"


def test_extract_unverified_catches_assumed():
    """ASSUMED_AWAITING_VERIFICATION is unverified per design memo §2.2."""
    rec = _make_record(adv_verdict_dicts=(
        {
            "assumption": "Y",
            "classification": "CARGO-CULTED",
            "rationale": "No source",
            "empirical_verification_status": EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
        },
    ))
    unverified = extract_unverified_assumptions(rec)
    assert len(unverified) == 1
    assert unverified[0].assumption == "Y"


def test_extract_unverified_legacy_row_treated_as_unverified():
    """Legacy rows lacking the field auto-classify INFERRED (= unverified)."""
    rec = _make_record(adv_verdict_dicts=(
        {"assumption": "Z", "classification": "CARGO-CULTED", "rationale": "Y"},
    ))
    unverified = extract_unverified_assumptions(rec)
    assert len(unverified) == 1
    assert unverified[0].empirical_verification_status == EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE


def test_extract_unverified_mixed_returns_only_unverified():
    """Mixed record returns only unverified entries."""
    rec = _make_record(adv_verdict_dicts=(
        {
            "assumption": "A",
            "classification": "HARD-EARNED",
            "rationale": "Z",
            "empirical_verification_status": EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            "evidence_artifact": "foo.py:1",
        },
        {
            "assumption": "B",
            "classification": "CARGO-CULTED",
            "rationale": "Z",
            "empirical_verification_status": EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
        },
        {
            "assumption": "C",
            "classification": "HARD-EARNED",
            "rationale": "Higham 2002",
            "empirical_verification_status": EmpiricalVerificationStatus.INFERRED_FROM_DOMAIN_LITERATURE,
        },
    ))
    unverified = extract_unverified_assumptions(rec)
    assert {ae.assumption for ae in unverified} == {"B", "C"}


# ──────────────────────────────────────────────────────────────────────
# verdict_status_requires_provisional_marker Round 3 predicate
# ──────────────────────────────────────────────────────────────────────


def test_provisional_marker_required_when_any_unverified():
    """One unverified assumption is sufficient for PROVISIONAL marker."""
    rec = _make_record(adv_verdict_dicts=(
        {
            "assumption": "X",
            "classification": "CARGO-CULTED",
            "rationale": "Z",
            "empirical_verification_status": EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
        },
    ))
    assert verdict_status_requires_provisional_marker(rec) is True


def test_provisional_marker_not_required_when_all_verified():
    """All-VERIFIED record does NOT need PROVISIONAL marker."""
    rec = _make_record(adv_verdict_dicts=(
        {
            "assumption": "X",
            "classification": "HARD-EARNED",
            "rationale": "Z",
            "empirical_verification_status": EmpiricalVerificationStatus.VERIFIED_VIA_SOURCE_INSPECTION,
            "evidence_artifact": "foo.py:1",
        },
    ))
    assert verdict_status_requires_provisional_marker(rec) is False


def test_provisional_marker_exempt_for_t1():
    """T1 working-group recommendations are exempt (design memo §2.1)."""
    # T1 records don't require adversary verdict per existing _validate_record.
    rec = CouncilDeliberationRecord(
        deliberation_id="t1_test",
        topic="t1",
        council_tier=CouncilTier.T1,
        council_attendees=("Yousfi",),
        council_quorum_met=True,
        council_verdict="PROCEED",
        council_assumption_adversary_verdict=(
            {
                "assumption": "X",
                "classification": "CARGO-CULTED",
                "rationale": "Z",
                "empirical_verification_status": EmpiricalVerificationStatus.ASSUMED_AWAITING_VERIFICATION,
            },
        ),
        # T1 has no mission_contribution requirement.
    )
    # T1 exempt even with unverified assumption.
    assert verdict_status_requires_provisional_marker(rec) is False


# ──────────────────────────────────────────────────────────────────────
# query_self_reflection_history_for_deliberation operator audit helper
# ──────────────────────────────────────────────────────────────────────


def test_query_self_reflection_history_empty_when_no_anchors(tmp_path: Path):
    """Empty store returns []."""
    posterior = tmp_path / "post.jsonl"
    result = query_self_reflection_history_for_deliberation(
        "any_id", posterior_path=posterior
    )
    assert result == []


def test_query_self_reflection_history_empty_when_no_match(tmp_path: Path):
    """No deliberation_id match returns []."""
    posterior = tmp_path / "post.jsonl"
    lock = tmp_path / "post.lock"
    rec = CouncilDeliberationRecord(
        deliberation_id="other_deliberation",
        topic="t",
        council_tier=CouncilTier.T2,
        council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
        council_quorum_met=True,
        council_verdict="PROCEED",
        council_assumption_adversary_verdict=(
            {
                "assumption": "X",
                "classification": "HARD-EARNED",
                "rationale": "Z",
            },
        ),
        event_type="council_self_reflection_round_1",
        predicted_mission_contribution="apparatus_maintenance",
    )
    append_council_anchor(rec, posterior_path=posterior, lock_path=lock)
    result = query_self_reflection_history_for_deliberation(
        "target_id", posterior_path=posterior
    )
    assert result == []


def test_query_self_reflection_history_returns_chain(tmp_path: Path):
    """Multiple self-reflection rounds return chronologically ordered."""
    posterior = tmp_path / "post.jsonl"
    lock = tmp_path / "post.lock"
    for i in range(1, 4):
        rec = CouncilDeliberationRecord(
            deliberation_id="target_id",
            topic="t",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
            council_quorum_met=True,
            council_verdict="PROCEED",
            council_assumption_adversary_verdict=(
                {"assumption": "X", "classification": "HARD-EARNED", "rationale": f"round {i}"},
            ),
            event_type=f"council_self_reflection_round_{i}",
            predicted_mission_contribution="apparatus_maintenance",
            written_at_utc=f"2026-05-26T00:0{i}:00+00:00",
        )
        append_council_anchor(rec, posterior_path=posterior, lock_path=lock)
    result = query_self_reflection_history_for_deliberation(
        "target_id", posterior_path=posterior
    )
    assert len(result) == 3
    # Chronological ordering by written_at_utc.
    assert [r.event_type for r in result] == [
        "council_self_reflection_round_1",
        "council_self_reflection_round_2",
        "council_self_reflection_round_3",
    ]


def test_query_self_reflection_history_filters_non_self_reflection_rows(tmp_path: Path):
    """Only event_type starting with council_self_reflection_round_ matches."""
    posterior = tmp_path / "post.jsonl"
    lock = tmp_path / "post.lock"
    # 1 dispatched + 1 self_reflection round for same deliberation_id.
    rec_disp = CouncilDeliberationRecord(
        deliberation_id="target_id",
        topic="t",
        council_tier=CouncilTier.T2,
        council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
        council_quorum_met=True,
        council_verdict="PROCEED",
        council_assumption_adversary_verdict=(
            {"assumption": "X", "classification": "HARD-EARNED", "rationale": "dispatch"},
        ),
        event_type="dispatched",
        predicted_mission_contribution="apparatus_maintenance",
    )
    rec_sr = CouncilDeliberationRecord(
        deliberation_id="target_id",
        topic="t",
        council_tier=CouncilTier.T2,
        council_attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
        council_quorum_met=True,
        council_verdict="PROCEED",
        council_assumption_adversary_verdict=(
            {"assumption": "X", "classification": "HARD-EARNED", "rationale": "round 1"},
        ),
        event_type="council_self_reflection_round_1",
        predicted_mission_contribution="apparatus_maintenance",
    )
    append_council_anchor(rec_disp, posterior_path=posterior, lock_path=lock)
    append_council_anchor(rec_sr, posterior_path=posterior, lock_path=lock)
    result = query_self_reflection_history_for_deliberation(
        "target_id", posterior_path=posterior
    )
    assert len(result) == 1
    assert result[0].event_type == "council_self_reflection_round_1"


def test_query_self_reflection_history_empty_deliberation_id_returns_empty():
    """Empty deliberation_id returns [] without scanning."""
    result = query_self_reflection_history_for_deliberation("")
    assert result == []
