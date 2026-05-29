# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_autopilot.canonical_invariants (GAP 4)."""
from __future__ import annotations

import pytest

from tac.cathedral_autopilot.canonical_invariants import (
    INVARIANT_NO_AD_HOC,
    INVARIANT_NO_DRIFT,
    INVARIANT_NO_DUPLICATE_CODE,
    INVARIANT_NO_REDISCOVERY,
    INVARIANT_NO_SIGNAL_LOSS,
    InvariantValidationStatus,
    InvariantValidationVerdict,
    VALID_INVARIANTS,
    validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants,
)


# ---------------------------------------------------------------------------
# Constants + enum
# ---------------------------------------------------------------------------


def test_5_canonical_invariants_pinned() -> None:
    assert len(VALID_INVARIANTS) == 5
    for inv in (
        INVARIANT_NO_AD_HOC,
        INVARIANT_NO_SIGNAL_LOSS,
        INVARIANT_NO_REDISCOVERY,
        INVARIANT_NO_DUPLICATE_CODE,
        INVARIANT_NO_DRIFT,
    ):
        assert inv in VALID_INVARIANTS


def test_status_enum_3_members() -> None:
    members = list(InvariantValidationStatus)
    assert len(members) == 3
    assert InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED in members
    assert InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED in members
    assert InvariantValidationStatus.UNKNOWN_INSUFFICIENT_EVIDENCE in members


# ---------------------------------------------------------------------------
# InvariantValidationVerdict invariants
# ---------------------------------------------------------------------------


def _all_pass_status() -> dict[str, str]:
    return {inv: "pass" for inv in VALID_INVARIANTS}


def _all_pass_rationale() -> dict[str, str]:
    return {inv: "ok" for inv in VALID_INVARIANTS}


def test_verdict_refuses_missing_keys() -> None:
    with pytest.raises(ValueError, match="per_invariant_status missing keys"):
        InvariantValidationVerdict(
            status=InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED,
            per_invariant_status={INVARIANT_NO_AD_HOC: "pass"},
            per_invariant_rationale={INVARIANT_NO_AD_HOC: "ok"},
            violated_invariants=(),
            rationale="x",
        )


def test_verdict_refuses_invalid_per_invariant_status() -> None:
    per_status = _all_pass_status()
    per_status[INVARIANT_NO_AD_HOC] = "bogus"
    with pytest.raises(ValueError, match="per_invariant_status"):
        InvariantValidationVerdict(
            status=InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED,
            per_invariant_status=per_status,
            per_invariant_rationale=_all_pass_rationale(),
            violated_invariants=(),
            rationale="x",
        )


def test_verdict_refuses_status_inconsistent_with_violations() -> None:
    """Cross-validation: status must match derived violations."""
    per_status = _all_pass_status()
    per_status[INVARIANT_NO_AD_HOC] = "fail"
    with pytest.raises(ValueError, match="must be FAIL"):
        InvariantValidationVerdict(
            status=InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED,  # WRONG
            per_invariant_status=per_status,
            per_invariant_rationale=_all_pass_rationale(),
            violated_invariants=(INVARIANT_NO_AD_HOC,),
            rationale="x",
        )


def test_verdict_refuses_violated_mismatch() -> None:
    per_status = _all_pass_status()
    per_status[INVARIANT_NO_AD_HOC] = "fail"
    with pytest.raises(ValueError, match="violated_invariants must match"):
        InvariantValidationVerdict(
            status=InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED,
            per_invariant_status=per_status,
            per_invariant_rationale=_all_pass_rationale(),
            violated_invariants=(INVARIANT_NO_DRIFT,),  # WRONG
            rationale="x",
        )


def test_verdict_refuses_promotable_true() -> None:
    with pytest.raises(ValueError, match="promotable MUST be False"):
        InvariantValidationVerdict(
            status=InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED,
            per_invariant_status=_all_pass_status(),
            per_invariant_rationale=_all_pass_rationale(),
            violated_invariants=(),
            rationale="x",
            promotable=True,
        )


def test_verdict_as_dict_round_trip() -> None:
    v = InvariantValidationVerdict(
        status=InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED,
        per_invariant_status=_all_pass_status(),
        per_invariant_rationale=_all_pass_rationale(),
        violated_invariants=(),
        rationale="x",
    )
    d = v.as_dict()
    assert d["status"] == "pass_all_invariants_honored"
    assert d["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# validate_* end-to-end
# ---------------------------------------------------------------------------


def test_validate_returns_unknown_on_empty_decision() -> None:
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({})
    assert v.status == InvariantValidationStatus.UNKNOWN_INSUFFICIENT_EVIDENCE
    assert len(v.violated_invariants) == 0


def test_validate_returns_pass_on_all_compliant_decision() -> None:
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({
        "ranking_via_canonical_three_metric_trichotomy": True,
        "pending_operator_corrections_count": 0,
        "canonical_anti_pattern_registry_consulted": True,
        "matched_anti_pattern_ids": (),
        "spawn_prompt_uses_canonical_helper": True,
        "duplicates_existing_canonical_module": False,
        "per_turn_ranking_deterministic_per_canonical_helper": True,
    })
    assert v.status == InvariantValidationStatus.PASS_ALL_INVARIANTS_HONORED
    assert v.violated_invariants == ()


def test_validate_flags_no_ad_hoc_violation() -> None:
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({
        "ranking_via_canonical_three_metric_trichotomy": False,
        "pending_operator_corrections_count": 0,
        "canonical_anti_pattern_registry_consulted": True,
        "matched_anti_pattern_ids": (),
        "spawn_prompt_uses_canonical_helper": True,
        "duplicates_existing_canonical_module": False,
        "per_turn_ranking_deterministic_per_canonical_helper": True,
    })
    assert v.status == InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED
    assert INVARIANT_NO_AD_HOC in v.violated_invariants


def test_validate_flags_no_signal_loss_violation() -> None:
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({
        "ranking_via_canonical_three_metric_trichotomy": True,
        "pending_operator_corrections_count": 3,
        "operator_correction_canonical_mutations_registered_this_turn": 0,
        "canonical_anti_pattern_registry_consulted": True,
        "matched_anti_pattern_ids": (),
        "spawn_prompt_uses_canonical_helper": True,
        "duplicates_existing_canonical_module": False,
        "per_turn_ranking_deterministic_per_canonical_helper": True,
    })
    assert v.status == InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED
    assert INVARIANT_NO_SIGNAL_LOSS in v.violated_invariants


def test_validate_flags_no_rediscovery_violation_when_matched_not_acknowledged() -> None:
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({
        "ranking_via_canonical_three_metric_trichotomy": True,
        "pending_operator_corrections_count": 0,
        "canonical_anti_pattern_registry_consulted": True,
        "matched_anti_pattern_ids": ("foo_v1",),
        "matched_anti_patterns_acknowledged": False,
        "spawn_prompt_uses_canonical_helper": True,
        "duplicates_existing_canonical_module": False,
        "per_turn_ranking_deterministic_per_canonical_helper": True,
    })
    assert v.status == InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED
    assert INVARIANT_NO_REDISCOVERY in v.violated_invariants


def test_validate_flags_no_duplicate_code_when_duplicates_existing() -> None:
    """Operator-empirically-caught canonical anti-pattern (today's incident)."""
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({
        "ranking_via_canonical_three_metric_trichotomy": True,
        "pending_operator_corrections_count": 0,
        "canonical_anti_pattern_registry_consulted": True,
        "matched_anti_pattern_ids": (),
        "spawn_prompt_uses_canonical_helper": True,
        "duplicates_existing_canonical_module": True,  # operator-caught case
        "per_turn_ranking_deterministic_per_canonical_helper": True,
    })
    assert v.status == InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED
    assert INVARIANT_NO_DUPLICATE_CODE in v.violated_invariants


def test_validate_flags_no_drift_violation() -> None:
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({
        "ranking_via_canonical_three_metric_trichotomy": True,
        "pending_operator_corrections_count": 0,
        "canonical_anti_pattern_registry_consulted": True,
        "matched_anti_pattern_ids": (),
        "spawn_prompt_uses_canonical_helper": True,
        "duplicates_existing_canonical_module": False,
        "per_turn_ranking_deterministic_per_canonical_helper": False,
    })
    assert v.status == InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED
    assert INVARIANT_NO_DRIFT in v.violated_invariants


def test_validate_multiple_violations_aggregated() -> None:
    v = validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants({
        "ranking_via_canonical_three_metric_trichotomy": False,
        "pending_operator_corrections_count": 3,
        "operator_correction_canonical_mutations_registered_this_turn": 0,
        "canonical_anti_pattern_registry_consulted": False,
        "matched_anti_pattern_ids": (),
        "spawn_prompt_uses_canonical_helper": False,
        "duplicates_existing_canonical_module": True,
        "per_turn_ranking_deterministic_per_canonical_helper": False,
    })
    assert v.status == InvariantValidationStatus.FAIL_ONE_OR_MORE_INVARIANTS_VIOLATED
    # All 5 should be violated
    assert len(v.violated_invariants) == 5


def test_validate_refuses_non_mapping_decision() -> None:
    with pytest.raises(ValueError, match="main_thread_decision must be a Mapping"):
        validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants(
            "not_a_mapping"
        )


def test_validate_refuses_non_mapping_canonical_state() -> None:
    with pytest.raises(ValueError, match="canonical_state must be a Mapping"):
        validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants(
            {}, canonical_state="not_a_mapping"
        )
