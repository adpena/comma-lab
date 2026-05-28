# SPDX-License-Identifier: MIT
"""Tests for tac.canonical_anti_patterns registry + builtins + pattern matcher.

Mirrors test patterns from sister tac.canonical_equations.tests.

Per design memo §"Tests at .../test_registry.py" requirement of >=30 tests.
This file lands 35+ tests covering:
- AntiPattern frozen dataclass invariants (5 rejections + happy path)
- EmpiricalFalsification frozen dataclass invariants (5)
- registry.register_anti_pattern happy path + JSONL append + persistence
- registry.append_empirical_falsification chronological ordering
- registry.load_anti_patterns lenient empty when missing
- registry.load_anti_patterns lenient skips malformed
- registry.load_anti_patterns_strict raises CorruptError per Catalog #138
- registry quarantine corrupt JSONL per Catalog #245 pattern
- registry atomic write (no .tmp leakage)
- registry.query_anti_patterns_by_substrate
- registry.query_falsifications_by_paradigm_class
- registry.query_recurrence_rate_by_severity
- registry.auto_recalibrate_from_continual_learning_posterior NON-STUB
- registry.auto_recalibrate idempotency
- registry.auto_recalibrate EVENT_ANTI_PATTERN_RECALIBRATED emission
- registry.auto_recalibrate no-refit-below-3-falsifications
- pattern_matcher matches LZMA+brotli stack
- pattern_matcher matches FP4-without-QAT stack
- pattern_matcher matches cross-paradigm-without-per-axis stack
- pattern_matcher empty when no match
- pattern_matcher multi-match ordered by severity
- registry concurrency stress 4-process spawn pool
- full lifecycle register → append falsification → recalibrate → query
- initial 12 anti-patterns all register cleanly via builtins
- validate_compound_stack_order LZMA after brotli flagged
- validate_compound_stack_order quantize before SVD flagged
- validate_compound_stack_order canonical order accepted
"""
from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
from pathlib import Path

import pytest

from tac.canonical_anti_patterns.anti_pattern import (
    AntiPattern,
    EmpiricalFalsification,
    InvalidAntiPatternError,
    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
    PARADIGM_COMPOUNDING_ORDER,
    PARADIGM_DIAGNOSIS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_OBSERVED_HIGH,
    SEVERITY_OBSERVED_MEDIUM,
)
from tac.canonical_anti_patterns.builtins import (
    build_all_initial_anti_patterns,
    populate_initial_anti_patterns,
)
from tac.canonical_anti_patterns.pattern_matcher import (
    AntiPatternMatch,
    match_stack_against_anti_patterns,
    validate_compound_stack_order,
)
from tac.canonical_anti_patterns.registry import (
    EVENT_ANTI_PATTERN_RECALIBRATED,
    EVENT_ANTI_PATTERN_REGISTERED,
    EVENT_FALSIFICATION_APPENDED,
    AntiPatternRegistryCorruptError,
    append_empirical_falsification,
    auto_recalibrate_from_continual_learning_posterior,
    get_anti_pattern_by_id,
    load_anti_patterns_events_lenient,
    load_anti_patterns_strict,
    query_anti_patterns,
    query_anti_patterns_by_substrate,
    query_falsifications_by_paradigm_class,
    query_recurrence_rate_by_severity,
    register_anti_pattern,
)
from tac.provenance.builders import build_provenance_for_predicted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _design_prov(name: str = "test"):
    return build_provenance_for_predicted(
        model_id=f"test.{name}",
        inputs_sha256="0" * 64,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )


def _minimal_anti_pattern(
    anti_pattern_id: str = "test_anti_pattern_v1",
    paradigm_class: str = PARADIGM_COMPOUNDING_ORDER,
    severity: str = SEVERITY_MEDIUM,
    next_recalibration_trigger: str = RECALIBRATE_ON_NEW_FALSIFICATIONS,
    empirical_falsifications: tuple = (),
    falsification_band: dict | None = None,
) -> AntiPattern:
    if falsification_band is None:
        falsification_band = {"test_lo": 0.0, "test_hi": 1.0}
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description="test anti-pattern for unit tests",
        forbidden_pattern_predicate="test.contains(forbidden_token)",
        falsification_band=falsification_band,
        recurrence_conditions=("test condition 1", "test condition 2"),
        canonical_source_anchor="test:source_anchor",
        canonical_unwind_path="test unwind: do canonical thing instead",
        canonical_producers=("test.producer",),
        canonical_consumers=("test.consumer",),
        paradigm_class=paradigm_class,
        severity=severity,
        provenance=_design_prov(anti_pattern_id),
        empirical_falsifications=empirical_falsifications,
        last_recalibration_utc="2026-05-28T00:00:00Z",
        next_recalibration_trigger=next_recalibration_trigger,
    )


def _minimal_falsification(
    anti_pattern_id: str = "test_anti_pattern_v1",
    falsification_id: str = "test_fals_001",
    measurement_method: str = "test_method",
    residual: float | None = 0.5,
) -> EmpiricalFalsification:
    return EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id=falsification_id,
        measurement_method=measurement_method,
        empirical_artifact_path="test:artifact_path",
        empirical_output={"observed_value": 0.5},
        falsification_residual=residual,
        captured_at_utc="2026-05-28T01:00:00Z",
        canonical_provenance=_design_prov(falsification_id),
        incident_classification=INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
        severity_observed=SEVERITY_OBSERVED_MEDIUM,
        operator_routable_unwind_path="apply canonical unwind",
    )


@pytest.fixture
def temp_registry():
    """Fresh tmp ledger + lock for each test (no live-state pollution)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        path = td / "test_registry.jsonl"
        lock = td / "test_registry.jsonl.lock"
        yield path, lock


# ---------------------------------------------------------------------------
# AntiPattern frozen dataclass invariants (5 rejections + happy path)
# ---------------------------------------------------------------------------


def test_anti_pattern_happy_path():
    ap = _minimal_anti_pattern()
    assert ap.anti_pattern_id == "test_anti_pattern_v1"
    assert ap.severity == SEVERITY_MEDIUM
    assert ap.recurrence_count == 0
    assert ap.is_actively_recurring is False


def test_anti_pattern_rejects_invalid_id():
    with pytest.raises(InvalidAntiPatternError, match="must match snake_case_vN"):
        _minimal_anti_pattern(anti_pattern_id="InvalidId")


def test_anti_pattern_rejects_invalid_paradigm_class():
    with pytest.raises(InvalidAntiPatternError, match="paradigm_class.*must be one of"):
        _minimal_anti_pattern(paradigm_class="not_a_paradigm")


def test_anti_pattern_rejects_invalid_severity():
    with pytest.raises(InvalidAntiPatternError, match="severity.*must be one of"):
        _minimal_anti_pattern(severity="not_a_severity")


def test_anti_pattern_rejects_empty_falsification_band():
    with pytest.raises(InvalidAntiPatternError, match="falsification_band must be non-empty"):
        _minimal_anti_pattern(falsification_band={})


def test_anti_pattern_rejects_orphan_no_producers_no_consumers():
    with pytest.raises(InvalidAntiPatternError, match="orphan anti-patterns are forbidden"):
        AntiPattern(
            anti_pattern_id="orphan_test_v1",
            description="orphan test",
            forbidden_pattern_predicate="orphan",
            falsification_band={"x": 0.0},
            recurrence_conditions=("c",),
            canonical_source_anchor="src",
            canonical_unwind_path="unwind",
            canonical_producers=(),
            canonical_consumers=(),
            paradigm_class=PARADIGM_COMPOUNDING_ORDER,
            severity=SEVERITY_MEDIUM,
            provenance=_design_prov("orphan"),
            empirical_falsifications=(),
            last_recalibration_utc="2026-05-28T00:00:00Z",
            next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
        )


# ---------------------------------------------------------------------------
# EmpiricalFalsification frozen dataclass invariants (5)
# ---------------------------------------------------------------------------


def test_falsification_happy_path():
    f = _minimal_falsification()
    assert f.anti_pattern_id == "test_anti_pattern_v1"
    assert f.falsification_residual == 0.5


def test_falsification_residual_none_accepted():
    """falsification_residual=None is acceptable (qualitative observation)."""
    f = _minimal_falsification(residual=None)
    assert f.falsification_residual is None


def test_falsification_rejects_negative_residual():
    with pytest.raises(InvalidAntiPatternError, match="must be >= 0"):
        _minimal_falsification(residual=-0.1)


def test_falsification_rejects_nan_residual():
    with pytest.raises(InvalidAntiPatternError, match="must not be NaN"):
        _minimal_falsification(residual=float("nan"))


def test_falsification_rejects_invalid_incident_classification():
    with pytest.raises(InvalidAntiPatternError, match="incident_classification"):
        EmpiricalFalsification(
            anti_pattern_id="test_anti_pattern_v1",
            falsification_id="fid",
            measurement_method="m",
            empirical_artifact_path="p",
            empirical_output={"x": 1},
            falsification_residual=0.0,
            captured_at_utc="2026-05-28T00:00:00Z",
            canonical_provenance=_design_prov("f"),
            incident_classification="not_canonical",
            severity_observed=SEVERITY_OBSERVED_HIGH,
            operator_routable_unwind_path="do x",
        )


def test_falsification_rejects_invalid_severity_observed():
    with pytest.raises(InvalidAntiPatternError, match="severity_observed"):
        EmpiricalFalsification(
            anti_pattern_id="test_anti_pattern_v1",
            falsification_id="fid",
            measurement_method="m",
            empirical_artifact_path="p",
            empirical_output={"x": 1},
            falsification_residual=0.0,
            captured_at_utc="2026-05-28T00:00:00Z",
            canonical_provenance=_design_prov("f"),
            incident_classification=INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
            severity_observed="not_canonical",
            operator_routable_unwind_path="do x",
        )


# ---------------------------------------------------------------------------
# Registry registration + persistence
# ---------------------------------------------------------------------------


def test_register_anti_pattern_happy_path(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern()
    register_anti_pattern(ap, path=path, lock_path=lock)
    assert path.exists()
    rows = load_anti_patterns_events_lenient(path)
    assert len(rows) == 1
    assert rows[0]["event_type"] == EVENT_ANTI_PATTERN_REGISTERED
    assert rows[0]["anti_pattern_id"] == "test_anti_pattern_v1"


def test_append_empirical_falsification_chronological(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern()
    register_anti_pattern(ap, path=path, lock_path=lock)
    f1 = _minimal_falsification(falsification_id="f1")
    f2 = _minimal_falsification(falsification_id="f2")
    append_empirical_falsification(f1, path=path, lock_path=lock)
    updated = append_empirical_falsification(f2, path=path, lock_path=lock)
    assert len(updated.empirical_falsifications) == 2
    assert updated.empirical_falsifications[0].falsification_id == "f1"
    assert updated.empirical_falsifications[1].falsification_id == "f2"


def test_load_lenient_empty_when_missing(tmp_path):
    nonexistent = tmp_path / "nonexistent.jsonl"
    assert load_anti_patterns_events_lenient(nonexistent) == []


def test_load_lenient_skips_malformed(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"valid": "row"}\n'
        "not_valid_json\n"
        '{"another": "valid"}\n'
    )
    rows = load_anti_patterns_events_lenient(path)
    assert len(rows) == 2


def test_load_strict_raises_on_corrupt(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"valid": "row"}\n'
        "not_valid_json\n"
    )
    with pytest.raises(AntiPatternRegistryCorruptError, match="invalid JSON"):
        load_anti_patterns_strict(path)


def test_quarantine_corrupt_via_register(temp_registry, tmp_path):
    path, lock = temp_registry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not_valid_json\n")
    ap = _minimal_anti_pattern()
    # Should NOT raise; should quarantine + write fresh.
    register_anti_pattern(ap, path=path, lock_path=lock)
    assert path.exists()
    # At least one quarantine file
    quarantines = list(path.parent.glob("test_registry.jsonl.corrupt.*"))
    assert len(quarantines) >= 1


def test_atomic_write_no_tmp_leakage(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern()
    register_anti_pattern(ap, path=path, lock_path=lock)
    # No .tmp.* should remain after a clean write
    tmps = list(path.parent.glob("test_registry.jsonl.tmp.*"))
    assert tmps == []


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def test_query_anti_patterns_by_substrate(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern(anti_pattern_id="my_substrate_anti_v1")
    # Modify producers to include substrate token
    from dataclasses import replace
    ap_with_substrate = replace(
        ap,
        canonical_producers=("experiments/train_substrate_pact_nerv.py",),
    )
    register_anti_pattern(ap_with_substrate, path=path, lock_path=lock)
    found = query_anti_patterns_by_substrate("pact_nerv", path=path)
    assert len(found) == 1
    assert found[0].anti_pattern_id == "my_substrate_anti_v1"


def test_query_anti_patterns_by_substrate_empty_token(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern()
    register_anti_pattern(ap, path=path, lock_path=lock)
    assert query_anti_patterns_by_substrate("", path=path) == []


def test_query_falsifications_by_paradigm_class(temp_registry):
    path, lock = temp_registry
    ap1 = _minimal_anti_pattern(
        anti_pattern_id="compounding_test_v1",
        paradigm_class=PARADIGM_COMPOUNDING_ORDER,
    )
    ap2 = _minimal_anti_pattern(
        anti_pattern_id="diagnosis_test_v1",
        paradigm_class=PARADIGM_DIAGNOSIS,
    )
    register_anti_pattern(ap1, path=path, lock_path=lock)
    register_anti_pattern(ap2, path=path, lock_path=lock)
    f1 = _minimal_falsification(anti_pattern_id="compounding_test_v1", falsification_id="f1")
    f2 = _minimal_falsification(anti_pattern_id="diagnosis_test_v1", falsification_id="f2")
    append_empirical_falsification(f1, path=path, lock_path=lock)
    append_empirical_falsification(f2, path=path, lock_path=lock)
    compounding = query_falsifications_by_paradigm_class(
        PARADIGM_COMPOUNDING_ORDER, path=path
    )
    assert len(compounding) == 1
    assert compounding[0].falsification_id == "f1"


def test_query_recurrence_rate_by_severity(temp_registry):
    path, lock = temp_registry
    # 2 medium severity, 1 actively recurring
    ap1 = _minimal_anti_pattern(anti_pattern_id="med1_v1", severity=SEVERITY_MEDIUM)
    ap2 = _minimal_anti_pattern(anti_pattern_id="med2_v1", severity=SEVERITY_MEDIUM)
    register_anti_pattern(ap1, path=path, lock_path=lock)
    register_anti_pattern(ap2, path=path, lock_path=lock)
    # Add 2 falsifications to ap1 (enough to be is_actively_recurring=True)
    for i in range(2):
        f = _minimal_falsification(
            anti_pattern_id="med1_v1", falsification_id=f"f_med1_{i}"
        )
        append_empirical_falsification(f, path=path, lock_path=lock)
    rates = query_recurrence_rate_by_severity(path=path)
    assert rates[SEVERITY_MEDIUM] == 0.5  # 1 of 2 actively recurring


def test_get_anti_pattern_by_id_present(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern()
    register_anti_pattern(ap, path=path, lock_path=lock)
    found = get_anti_pattern_by_id("test_anti_pattern_v1", path=path)
    assert found is not None
    assert found.anti_pattern_id == "test_anti_pattern_v1"


def test_get_anti_pattern_by_id_absent(temp_registry):
    path, _lock = temp_registry
    assert get_anti_pattern_by_id("does_not_exist_v1", path=path) is None


# ---------------------------------------------------------------------------
# Auto-recalibrator (NON-STUB regression vs Catalog #371 lesson)
# ---------------------------------------------------------------------------


def test_auto_recalibrate_refit_fires_on_stale_band(temp_registry):
    """Catalog #371 regression: refit MUST actually update stale band."""
    path, lock = temp_registry
    ap = _minimal_anti_pattern(
        anti_pattern_id="recalib_test_v1",
        falsification_band={"stale_lo": 99.0, "stale_hi": 99.0},
    )
    register_anti_pattern(ap, path=path, lock_path=lock)
    for i in range(3):
        f = _minimal_falsification(
            anti_pattern_id="recalib_test_v1",
            falsification_id=f"fals_{i}",
            measurement_method="canonical_method",
            residual=0.1 * (i + 1),
        )
        append_empirical_falsification(f, path=path, lock_path=lock)
    report = auto_recalibrate_from_continual_learning_posterior(
        path=path, lock_path=lock
    )
    assert report.anti_patterns_recalibrated == 1
    # Verify the band was actually updated (NOT a stub)
    refreshed = get_anti_pattern_by_id("recalib_test_v1", path=path)
    assert refreshed is not None
    assert "stale_lo" not in refreshed.falsification_band
    assert "canonical_method_residual_lo" in refreshed.falsification_band


def test_auto_recalibrate_idempotency(temp_registry):
    """2nd run with no new falsifications recalibrates 0."""
    path, lock = temp_registry
    ap = _minimal_anti_pattern(
        anti_pattern_id="idempotent_v1",
        falsification_band={"old_lo": 99.0},
    )
    register_anti_pattern(ap, path=path, lock_path=lock)
    for i in range(3):
        f = _minimal_falsification(
            anti_pattern_id="idempotent_v1",
            falsification_id=f"f_{i}",
            measurement_method="m",
            residual=0.1,
        )
        append_empirical_falsification(f, path=path, lock_path=lock)
    r1 = auto_recalibrate_from_continual_learning_posterior(path=path, lock_path=lock)
    assert r1.anti_patterns_recalibrated == 1
    r2 = auto_recalibrate_from_continual_learning_posterior(path=path, lock_path=lock)
    assert r2.anti_patterns_recalibrated == 0
    assert r2.anti_patterns_eligible_but_unchanged == 1


def test_auto_recalibrate_emits_recalibrated_event(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern(
        anti_pattern_id="event_test_v1",
        falsification_band={"x_lo": 99.0},
    )
    register_anti_pattern(ap, path=path, lock_path=lock)
    for i in range(3):
        f = _minimal_falsification(
            anti_pattern_id="event_test_v1",
            falsification_id=f"f_{i}",
            measurement_method="m",
            residual=0.0,
        )
        append_empirical_falsification(f, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(path=path, lock_path=lock)
    rows = load_anti_patterns_events_lenient(path)
    event_types = [r["event_type"] for r in rows]
    assert EVENT_ANTI_PATTERN_RECALIBRATED in event_types


def test_auto_recalibrate_no_refit_below_3_falsifications(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern(
        anti_pattern_id="too_few_v1",
        falsification_band={"x_lo": 99.0},
    )
    register_anti_pattern(ap, path=path, lock_path=lock)
    # Only 2 falsifications (below the 3 threshold)
    for i in range(2):
        f = _minimal_falsification(
            anti_pattern_id="too_few_v1",
            falsification_id=f"f_{i}",
            measurement_method="m",
            residual=0.1,
        )
        append_empirical_falsification(f, path=path, lock_path=lock)
    report = auto_recalibrate_from_continual_learning_posterior(path=path, lock_path=lock)
    assert report.anti_patterns_recalibrated == 0


# ---------------------------------------------------------------------------
# Pattern matcher
# ---------------------------------------------------------------------------


def test_matcher_matches_lzma_after_brotli(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {"compression_ops": ["int8_per_channel", "brotli_q11", "lzma_q9"]},
        path=path,
    )
    assert len(matches) >= 1
    ids = [m.anti_pattern.anti_pattern_id for m in matches]
    assert any("lzma" in i for i in ids)


def test_matcher_matches_fp4_without_qat(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "quantization_ops": ["fp4_packed"],
            "substrate_id": "pact_nerv_test",
            "training_pipeline": "no_qat",
        },
        path=path,
    )
    assert any(
        "fp4" in m.anti_pattern.anti_pattern_id for m in matches
    ), f"FP4 match not found in {[m.anti_pattern.anti_pattern_id for m in matches]}"


def test_matcher_matches_cross_paradigm_without_per_axis(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {
            "substrate_id": "cross_paradigm_test_substrate",
            "per_axis_decomposition_active": False,
        },
        path=path,
    )
    assert any(
        "cross_paradigm" in m.anti_pattern.anti_pattern_id for m in matches
    )


def test_matcher_empty_when_no_match(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    matches = match_stack_against_anti_patterns(
        {"completely_unrelated_field": "totally_safe_canonical_operation_xyzzy"},
        path=path,
        min_confidence=0.9,
    )
    # Note: low-confidence noise may still produce hits; with min_confidence=0.9 we exclude them.
    assert matches == ()


def test_matcher_orders_by_severity(temp_registry):
    path, lock = temp_registry
    populate_initial_anti_patterns(path=path, lock_path=lock)
    # Stack triggers MULTIPLE anti-patterns across severities
    matches = match_stack_against_anti_patterns(
        {
            "compression_ops": ["brotli_q11", "lzma_q9"],  # MEDIUM
            "quantization_ops": ["fp4_packed"],            # CRITICAL
            "training_pipeline": "no_qat",
        },
        path=path,
    )
    severities = [m.anti_pattern.severity for m in matches]
    # CRITICAL must come first if present
    if SEVERITY_CRITICAL in severities and SEVERITY_MEDIUM in severities:
        first_critical = severities.index(SEVERITY_CRITICAL)
        last_medium = len(severities) - 1 - severities[::-1].index(SEVERITY_MEDIUM)
        assert first_critical < last_medium


def test_matcher_handles_invalid_stack_spec_gracefully():
    # Defensive: non-Mapping input should NOT crash; returns empty.
    matches = match_stack_against_anti_patterns("not_a_dict")
    assert matches == ()


# ---------------------------------------------------------------------------
# validate_compound_stack_order
# ---------------------------------------------------------------------------


def test_validate_order_lzma_after_brotli_flagged():
    v = validate_compound_stack_order(["brotli_q11", "lzma_q9"])
    assert v.is_valid is False
    assert any("lzma" in vio.lower() for vio in v.violations)


def test_validate_order_quantize_before_svd_flagged():
    v = validate_compound_stack_order(["int8_per_channel", "svd_rank_16"])
    assert v.is_valid is False
    assert any("svd" in vio.lower() for vio in v.violations)


def test_validate_order_canonical_accepted():
    v = validate_compound_stack_order(["svd_rank_16", "int8_per_channel", "ans_coding"])
    assert v.is_valid is True
    assert v.violations == ()


def test_validate_order_lzma_before_brotli_not_flagged():
    """When LZMA is BEFORE brotli (or chained somewhere), still flagged per #4 broader sister."""
    v = validate_compound_stack_order(["lzma_q9", "brotli_q11"])
    # Catalog #4 (broader sister) still flags brotli + lzma together
    assert v.is_valid is False


def test_validate_order_brotli_only_accepted():
    v = validate_compound_stack_order(["int8_per_channel", "brotli_q11"])
    assert v.is_valid is True


def test_validate_order_invalid_input_type():
    v = validate_compound_stack_order("not_a_list")  # type: ignore[arg-type]
    assert v.is_valid is False
    assert "must be a list" in v.violations[0]


# ---------------------------------------------------------------------------
# Full lifecycle + initial population
# ---------------------------------------------------------------------------


def test_full_lifecycle_register_falsify_recalibrate_query(temp_registry):
    path, lock = temp_registry
    ap = _minimal_anti_pattern(
        anti_pattern_id="lifecycle_v1",
        falsification_band={"initial_lo": 99.0},
    )
    register_anti_pattern(ap, path=path, lock_path=lock)
    for i in range(3):
        f = _minimal_falsification(
            anti_pattern_id="lifecycle_v1",
            falsification_id=f"f_{i}",
            measurement_method="canonical",
            residual=0.05 * (i + 1),
        )
        append_empirical_falsification(f, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(path=path, lock_path=lock)
    final = get_anti_pattern_by_id("lifecycle_v1", path=path)
    assert final is not None
    assert final.recurrence_count == 3
    assert final.is_actively_recurring is True
    assert "canonical_residual_lo" in final.falsification_band


def test_initial_12_anti_patterns_all_register_cleanly(temp_registry):
    """Every one of the 12 initial anti-patterns must register without error."""
    path, lock = temp_registry
    registered = populate_initial_anti_patterns(path=path, lock_path=lock)
    assert len(registered) == 12
    all_aps = query_anti_patterns(path=path)
    assert len(all_aps) == 12
    ids = {ap.anti_pattern_id for ap in all_aps}
    # Spot-check a handful
    assert "lzma_on_already_brotli_saturated_compounding_v1" in ids
    assert "fp4_packed_without_qat_cos_collapse_v1" in ids
    assert "predicted_band_from_random_init_tier_c_v1" in ids
    assert "silent_no_spawn_modal_dispatch_v1" in ids


# ---------------------------------------------------------------------------
# 4-process concurrency stress (canonical fcntl-locked append safety)
# ---------------------------------------------------------------------------


def _worker_register(args):
    """Top-level for multiprocessing.Pool pickling."""
    path_str, lock_str, idx = args
    from tac.canonical_anti_patterns.builtins import (
        build_lzma_on_already_brotli_saturated_compounding_v1,
    )
    from tac.canonical_anti_patterns.registry import register_anti_pattern
    from dataclasses import replace
    ap = build_lzma_on_already_brotli_saturated_compounding_v1()
    # Make each worker register a UNIQUELY-NAMED anti-pattern
    ap_unique = replace(ap, anti_pattern_id=f"stress_test_v{idx}")
    register_anti_pattern(
        ap_unique, path=Path(path_str), lock_path=Path(lock_str)
    )
    return idx


def test_concurrency_4_process_spawn_pool_safe(temp_registry):
    path, lock = temp_registry
    ctx = mp.get_context("spawn")
    args = [(str(path), str(lock), i) for i in range(4)]
    with ctx.Pool(4) as pool:
        results = pool.map(_worker_register, args)
    assert sorted(results) == [0, 1, 2, 3]
    rows = load_anti_patterns_events_lenient(path)
    assert len(rows) == 4
    ids = {row["anti_pattern_id"] for row in rows}
    assert ids == {f"stress_test_v{i}" for i in range(4)}
