# SPDX-License-Identifier: MIT
"""Tests for Slot A NEGATIVE-RESULTS-AUDIT-V2 TOP-3 op-routables FIX O1 + O2.

Per Slot C canonical-2-landing-pattern corrective-action cascade 2026-05-28:

  * **FIX O1** (Catalog #371 recalibrator excluded_contexts fix) — anchors whose
    ``inputs.in_domain_context`` matches the equation's
    ``domain_of_validity.excluded_contexts`` are SKIPPED when refitting the
    residual SUMMARY map. The anchor row itself remains in the registry per
    Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (it IS real empirical
    evidence for an OUT-OF-DOMAIN context); only the residual SUMMARY excludes
    it. Sister of Catalog #359 (residual-hybrid misapplication) at the
    per-anchor recalibration surface.

  * **FIX O2** (Catalog #371 recalibrator NaN-sentinel skip) — anchors with
    NaN/inf residuals are SKIPPED. Defense-in-depth:
    ``EmpiricalAnchor.__post_init__`` already refuses NaN at construction.
    Additionally, equations with NaN/inf in their STORED residual map (legacy
    payloads predating the construction invariant) qualify for the secondary
    NaN-cleanup-eligibility path even when below the 3-anchor threshold.

Empirical anchors per Slot A audit memo
``feedback_negative_results_audit_v2_mathematical_groundedness_plus_optimality_plus_canonical_apparatus_mutation_gaps_landed_20260528.md``:

  * F05: ``pose_axis_score_direction_matching_paradigm_savings_v1`` anchor
    with ``in_domain_context='mlx_native_pose_axis_score_direction_matching_standalone_replaces_segnet'``
    produced residual=30.68 which dominated the residual map even though the
    standalone-replace context IS in ``excluded_contexts``.

  * F08+F09: ``mlx_cuda_bidirectional_drift_engineering_response_v1`` +
    ``daubechies_multi_scale_wavelet_hierarchical_composition_savings_v1``
    have NaN entries in their stored residual map.
"""
from __future__ import annotations

import math
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.registry import (
    EVENT_RECALIBRATED,
    _anchor_in_domain_context,
    _anchor_is_in_excluded_context,
    _anchor_residual_is_nan_or_infinite,
    _refit_residual_map_from_anchors,
    _stored_map_has_corrupt_residual_keys,
    auto_recalibrate_from_continual_learning_posterior,
    get_equation_by_id,
    load_registry_events_lenient,
    register_canonical_equation,
)
from tac.provenance.builders import build_provenance_for_predicted


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _prov():
    return build_provenance_for_predicted(
        model_id="test.slot_c.fix_o1_o2.v1",
        inputs_sha256="0" * 64,
    )


def _anchor(
    method: str,
    residual: float,
    *,
    in_domain_context: str | None = None,
    inputs: dict | None = None,
) -> EmpiricalAnchor:
    base_inputs = dict(inputs) if inputs else {"x": 1}
    if in_domain_context is not None:
        base_inputs["in_domain_context"] = in_domain_context
    return EmpiricalAnchor(
        anchor_id=f"anchor_{method}",
        measurement_utc="2026-05-28T00:00:00Z",
        inputs=base_inputs,
        predicted_output={"y": 0.15},
        empirical_output={"y": residual},
        residual=residual,
        source_artifact="test_artifact",
        measurement_method=method,
        provenance=_prov(),
    )


def _eq(
    eq_id: str,
    *,
    anchors: tuple[EmpiricalAnchor, ...] = (),
    residual_map: dict[str, float] | None = None,
    domain_of_validity: dict | None = None,
    trigger: str = RECALIBRATE_ON_NEW_ANCHORS,
) -> CanonicalEquation:
    return CanonicalEquation(
        equation_id=eq_id,
        name=eq_id.replace("_", " "),
        one_line_summary="test equation for slot c fix o1 o2",
        latex_form=r"\Delta S = \alpha",
        python_callable_module_path="tac.canonical_equations.tests.fixture:predict",
        domain_of_validity=domain_of_validity or {"test": True},
        units_in={"x": "unit"},
        units_out={"y": "unit"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual=residual_map or {},
        last_calibration_utc="2026-05-01T00:00:00Z",
        next_recalibration_trigger=trigger,
        canonical_consumers=("tac.test_consumer",),
        canonical_producers=("tac.test_producer",),
        provenance=_prov(),
    )


def _setup(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    return path, lock


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_anchor_in_domain_context_returns_string_or_none() -> None:
    """``_anchor_in_domain_context`` returns the canonical token or None."""
    with_ctx = _anchor("arm_a", 0.5, in_domain_context="in_domain_X")
    assert _anchor_in_domain_context(with_ctx) == "in_domain_X"

    without_ctx = _anchor("arm_b", 0.5)
    assert _anchor_in_domain_context(without_ctx) is None


def test_anchor_in_domain_context_strips_whitespace() -> None:
    """Trailing whitespace is stripped; empty string returns None."""
    spaced = _anchor("arm_a", 0.5, in_domain_context="  ctx_X  ")
    assert _anchor_in_domain_context(spaced) == "ctx_X"

    blank = _anchor("arm_b", 0.5, in_domain_context="   ")
    assert _anchor_in_domain_context(blank) is None


def test_anchor_in_domain_context_non_string_returns_none() -> None:
    """Non-string ``in_domain_context`` (e.g. dict / int / list) returns None."""
    int_ctx = _anchor("arm_a", 0.5, inputs={"in_domain_context": 42})
    assert _anchor_in_domain_context(int_ctx) is None


def test_anchor_is_in_excluded_context_matches_list() -> None:
    """Matches when context is in ``excluded_contexts`` list."""
    anchor = _anchor("arm_a", 0.5, in_domain_context="excluded_X")
    excluded = ("excluded_X", "excluded_Y")
    assert _anchor_is_in_excluded_context(anchor, excluded) is True


def test_anchor_is_in_excluded_context_no_match() -> None:
    """Does NOT match when context is NOT in ``excluded_contexts`` list."""
    anchor = _anchor("arm_a", 0.5, in_domain_context="included_Z")
    excluded = ("excluded_X", "excluded_Y")
    assert _anchor_is_in_excluded_context(anchor, excluded) is False


def test_anchor_is_in_excluded_context_empty_excluded_returns_false() -> None:
    """Empty/None ``excluded_contexts`` returns False (no exclusion active)."""
    anchor = _anchor("arm_a", 0.5, in_domain_context="any_ctx")
    assert _anchor_is_in_excluded_context(anchor, []) is False
    assert _anchor_is_in_excluded_context(anchor, None) is False


def test_anchor_is_in_excluded_context_no_in_domain_context_returns_false() -> None:
    """Anchor without ``in_domain_context`` returns False (cannot exclude)."""
    anchor = _anchor("arm_a", 0.5)  # no in_domain_context
    assert _anchor_is_in_excluded_context(anchor, ("excluded_X",)) is False


def test_anchor_residual_is_nan_or_infinite_detects_inf() -> None:
    """Pure-inf residual MAY land if a future loader-path slips it through;
    the canonical EmpiricalAnchor invariants refuse NaN at construction, but
    inf is currently allowed (inf >= 0 passes); the helper is defense-in-depth."""
    inf_anchor = _anchor("arm_a", float("inf"))
    assert _anchor_residual_is_nan_or_infinite(inf_anchor) is True


def test_anchor_residual_is_nan_or_infinite_normal_values_pass() -> None:
    """Normal float residuals (0.0, small positive, large positive) return False."""
    assert _anchor_residual_is_nan_or_infinite(_anchor("arm_a", 0.0)) is False
    assert _anchor_residual_is_nan_or_infinite(_anchor("arm_b", 0.5)) is False
    assert _anchor_residual_is_nan_or_infinite(_anchor("arm_c", 100.0)) is False


def test_stored_map_has_corrupt_residual_keys_detects_nan() -> None:
    """NaN in stored map is detected."""
    nan_map = {"axis_a": 0.5, "axis_b": float("nan")}
    assert _stored_map_has_corrupt_residual_keys(nan_map) is True


def test_stored_map_has_corrupt_residual_keys_detects_inf() -> None:
    """Infinite residual in stored map is detected."""
    inf_map = {"axis_a": 0.5, "axis_b": float("inf")}
    assert _stored_map_has_corrupt_residual_keys(inf_map) is True

    neg_inf_map = {"axis_a": 0.5, "axis_b": float("-inf")}
    assert _stored_map_has_corrupt_residual_keys(neg_inf_map) is True


def test_stored_map_has_corrupt_residual_keys_clean_returns_false() -> None:
    """Clean stored map (no NaN/inf) returns False."""
    clean_map = {"axis_a": 0.5, "axis_b": 1.5, "axis_c": 0.0}
    assert _stored_map_has_corrupt_residual_keys(clean_map) is False
    assert _stored_map_has_corrupt_residual_keys({}) is False


# ---------------------------------------------------------------------------
# FIX O1 — excluded-context filter behavior
# ---------------------------------------------------------------------------


def test_fix_o1_refit_excludes_anchors_in_excluded_contexts(tmp_path: Path) -> None:
    """F05 empirical anchor regression — anchor with in_domain_context in
    excluded_contexts must be DROPPED from the refit residual map."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "pose_axis_excluded_ctx_v1",
        anchors=(
            _anchor("arm_a", 0.0, in_domain_context="in_domain_ctx_A"),
            _anchor("arm_b", 0.5, in_domain_context="in_domain_ctx_A"),
            # F05 mirror: this anchor lives in an EXCLUDED context.
            _anchor("arm_c_excluded", 30.68, in_domain_context="standalone_excluded_X"),
        ),
        domain_of_validity={
            "test": True,
            "excluded_contexts": ["standalone_excluded_X"],
        },
        residual_map={"arm_a": 0.0, "arm_b": 0.5, "arm_c_excluded": 30.68},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "pose_axis_excluded_ctx_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("pose_axis_excluded_ctx_v1", path=path)
    assert after is not None
    # The excluded anchor's residual must NOT appear in the refit map.
    assert "arm_c_excluded" not in after.predicted_vs_empirical_residual
    # The non-excluded anchors remain.
    assert after.predicted_vs_empirical_residual["arm_a"] == 0.0
    assert after.predicted_vs_empirical_residual["arm_b"] == 0.5


def test_fix_o1_refit_preserves_anchor_in_registry_per_append_only(tmp_path: Path) -> None:
    """The excluded anchor remains in the empirical_anchors tuple (APPEND-ONLY);
    only the residual SUMMARY excludes it."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "pose_axis_preserve_anchor_v1",
        anchors=(
            _anchor("arm_a", 0.0, in_domain_context="in_domain_ctx_A"),
            _anchor("arm_b", 0.0, in_domain_context="in_domain_ctx_A"),
            _anchor("arm_c_excluded", 30.68, in_domain_context="standalone_excluded_X"),
        ),
        domain_of_validity={
            "test": True,
            "excluded_contexts": ["standalone_excluded_X"],
        },
        residual_map={"arm_a": 0.0, "arm_b": 0.0, "arm_c_excluded": 30.68},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(
        "pose_axis_preserve_anchor_v1", path=path, lock_path=lock
    )
    after = get_equation_by_id("pose_axis_preserve_anchor_v1", path=path)
    assert after is not None
    # APPEND-ONLY: all 3 anchors STILL present in the empirical_anchors tuple.
    assert len(after.empirical_anchors) == 3
    excluded_anchor = [
        a
        for a in after.empirical_anchors
        if a.inputs.get("in_domain_context") == "standalone_excluded_X"
    ]
    assert len(excluded_anchor) == 1
    assert excluded_anchor[0].residual == 30.68


def test_fix_o1_no_excluded_contexts_field_falls_back_to_legacy_refit(tmp_path: Path) -> None:
    """When the equation has NO ``excluded_contexts``, refit behavior is unchanged
    from legacy (all anchors land in the residual map)."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "no_excluded_ctx_v1",
        anchors=(
            _anchor("arm_a", 0.0, in_domain_context="ctx_A"),
            _anchor("arm_b", 0.5, in_domain_context="ctx_B"),
            _anchor("arm_c", 1.0, in_domain_context="ctx_C"),
        ),
        domain_of_validity={"test": True},  # no excluded_contexts
        residual_map={"arm_a": 999.0},  # stale prior
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "no_excluded_ctx_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("no_excluded_ctx_v1", path=path)
    assert after is not None
    # All 3 anchors land (none excluded; legacy behavior preserved).
    assert after.predicted_vs_empirical_residual == {
        "arm_a": 0.0,
        "arm_b": 0.5,
        "arm_c": 1.0,
    }


# ---------------------------------------------------------------------------
# FIX O2 — NaN-sentinel skip + secondary eligibility behavior
# ---------------------------------------------------------------------------


def test_fix_o2_nan_cleanup_secondary_eligibility_fires_below_3_anchors(tmp_path: Path) -> None:
    """F08+F09 empirical anchor regression — equation with 1 anchor + stored
    NaN-contaminated map qualifies for refit via the SECONDARY NaN-cleanup
    eligibility path even though it's below the 3-anchor threshold."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "nan_contaminated_below_3_v1",
        anchors=(_anchor("arm_a", 0.0),),
        residual_map={
            "arm_a": 0.0,
            "stale_axis_nan": float("nan"),
        },
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "nan_contaminated_below_3_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("nan_contaminated_below_3_v1", path=path)
    assert after is not None
    # NaN axis dropped; only the anchor-derived axis remains.
    assert after.predicted_vs_empirical_residual == {"arm_a": 0.0}
    assert "stale_axis_nan" not in after.predicted_vs_empirical_residual


def test_fix_o2_nan_cleanup_eligible_only_when_recalibrate_trigger_matches(tmp_path: Path) -> None:
    """Secondary NaN-cleanup eligibility requires the canonical
    RECALIBRATE_ON_NEW_ANCHORS trigger. Equations on NEVER_AUTO are NOT touched
    even when their stored map is NaN-contaminated."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "nan_never_auto_v1",
        anchors=(_anchor("arm_a", 0.0),),
        residual_map={"stale_nan_axis": float("nan")},
        trigger=RECALIBRATE_NEVER_AUTO,
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "nan_never_auto_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 0
    after = get_equation_by_id("nan_never_auto_v1", path=path)
    assert after is not None
    # Operator-only trigger: NaN preserved (operator must invoke explicit
    # recalibrate_equation for these).
    assert "stale_nan_axis" in after.predicted_vs_empirical_residual
    stored_nan = after.predicted_vs_empirical_residual["stale_nan_axis"]
    assert math.isnan(stored_nan)


def test_fix_o2_nan_cleanup_requires_at_least_one_anchor(tmp_path: Path) -> None:
    """Zero-anchor equations cannot refit (no evidence to refit FROM); the
    secondary eligibility path requires at least 1 landed anchor."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "nan_zero_anchors_v1",
        anchors=(),
        residual_map={"stale_nan_axis": float("nan")},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "nan_zero_anchors_v1", path=path, lock_path=lock
    )
    # Zero anchors: cannot refit; NaN preserved in summary (operator must
    # explicitly invoke recalibrate or land at least one anchor).
    assert rep.equations_recalibrated == 0


def test_fix_o2_well_calibrated_after_nan_cleanup(tmp_path: Path) -> None:
    """After NaN-cleanup, ``is_well_calibrated`` reflects the cleaned map
    rather than the NaN-poisoned legacy state."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "nan_well_calibrated_v1",
        anchors=(_anchor("arm_a", 0.1),),
        residual_map={"arm_a": 0.1, "stale_nan": float("nan")},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(
        "nan_well_calibrated_v1", path=path, lock_path=lock
    )
    after = get_equation_by_id("nan_well_calibrated_v1", path=path)
    assert after is not None
    # After cleanup, the only residual is 0.1 (< 2.0 well-calibrated threshold).
    assert after.is_well_calibrated is True


# ---------------------------------------------------------------------------
# Combined FIX O1 + O2 behavior
# ---------------------------------------------------------------------------


def test_combined_o1_o2_both_filters_apply(tmp_path: Path) -> None:
    """Both filters apply in the same refit: excluded-context anchor SKIPPED
    AND NaN-contaminated stored axis DROPPED."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "combined_o1_o2_v1",
        anchors=(
            _anchor("arm_a", 0.1, in_domain_context="in_domain_X"),
            _anchor("arm_b", 0.2, in_domain_context="in_domain_X"),
            _anchor("arm_c", 0.3, in_domain_context="in_domain_X"),
            _anchor("arm_d_excluded", 99.0, in_domain_context="excluded_Y"),
        ),
        domain_of_validity={
            "test": True,
            "excluded_contexts": ["excluded_Y"],
        },
        residual_map={
            "arm_a": 0.1,
            "arm_b": 0.2,
            "arm_c": 0.3,
            "arm_d_excluded": 99.0,
            "stale_nan_axis": float("nan"),
        },
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "combined_o1_o2_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("combined_o1_o2_v1", path=path)
    assert after is not None
    # Excluded anchor + NaN axis BOTH gone; in-domain anchors retained.
    assert after.predicted_vs_empirical_residual == {
        "arm_a": 0.1,
        "arm_b": 0.2,
        "arm_c": 0.3,
    }


# ---------------------------------------------------------------------------
# Canonical 4-step regression — APPEND-ONLY event emission
# ---------------------------------------------------------------------------


def test_recalibration_emits_event_recalibrated_row(tmp_path: Path) -> None:
    """Refit fires the canonical EVENT_RECALIBRATED row per APPEND-ONLY."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "event_recalibrated_emission_v1",
        anchors=(
            _anchor("arm_a", 0.0),
            _anchor("arm_b", 0.0),
            _anchor("arm_c", 0.0),
        ),
        residual_map={"stale_prior": 0.15},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(
        "event_recalibrated_emission_v1", path=path, lock_path=lock
    )
    rows = load_registry_events_lenient(path)
    recalibrated_rows = [r for r in rows if r.get("event_type") == EVENT_RECALIBRATED]
    assert len(recalibrated_rows) == 1
    # The recalibrated row's notes must mention the canonical FIX O1+O2 source.
    notes = recalibrated_rows[0].get("notes", "")
    assert "FIX O1+O2" in notes or "excluded_contexts + NaN" in notes


def test_recalibration_idempotency_after_o1_o2_landings(tmp_path: Path) -> None:
    """A second recalibration pass after FIX O1+O2 landed produces zero new
    recalibration (the system is now in canonical state)."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "idempotency_after_o1_o2_v1",
        anchors=(
            _anchor("arm_a", 0.1, in_domain_context="in_domain_X"),
            _anchor("arm_b", 0.2, in_domain_context="in_domain_X"),
            _anchor("arm_c", 0.3, in_domain_context="in_domain_X"),
            _anchor("arm_excl", 99.0, in_domain_context="excluded_Y"),
        ),
        domain_of_validity={"excluded_contexts": ["excluded_Y"]},
        residual_map={"arm_a": 0.1, "arm_b": 0.2, "arm_c": 0.3, "arm_excl": 99.0},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep1 = auto_recalibrate_from_continual_learning_posterior(
        "idempotency_after_o1_o2_v1", path=path, lock_path=lock
    )
    assert rep1.equations_recalibrated == 1
    # Second pass: already canonical, no refit fires.
    rep2 = auto_recalibrate_from_continual_learning_posterior(
        "idempotency_after_o1_o2_v1", path=path, lock_path=lock
    )
    assert rep2.equations_recalibrated == 0


def test_per_equation_summary_carries_nan_cleanup_eligibility_flag(tmp_path: Path) -> None:
    """RecalibrationReport.per_equation_summary carries the new
    ``nan_cleanup_eligible`` flag so the operator-facing summary discloses
    the secondary path explicitly."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "summary_flag_v1",
        anchors=(_anchor("arm_a", 0.0),),
        residual_map={"stale_nan": float("nan")},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "summary_flag_v1", path=path, lock_path=lock
    )
    summary = rep.per_equation_summary["summary_flag_v1"]
    assert "nan_cleanup_eligible" in summary
    assert summary["nan_cleanup_eligible"] is True
    assert summary["refit_eligible"] is False  # below 3-anchor threshold


def test_refit_does_not_synthesize_anchors(tmp_path: Path) -> None:
    """Per Catalog #287 + #323: refit MUST NOT add new anchors to the equation;
    only the residual summary is re-derived. APPEND-ONLY preserved."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "no_synthesis_v1",
        anchors=(
            _anchor("arm_a", 0.0),
            _anchor("arm_b", 0.5),
            _anchor("arm_c", 1.0),
        ),
        residual_map={"stale_prior": 0.15},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(
        "no_synthesis_v1", path=path, lock_path=lock
    )
    after = get_equation_by_id("no_synthesis_v1", path=path)
    assert after is not None
    # Anchor count unchanged: 3 in, 3 out.
    assert len(after.empirical_anchors) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_refit_when_all_anchors_excluded_produces_empty_residual_map(tmp_path: Path) -> None:
    """Edge case: every anchor is in an excluded context. Refit produces an
    empty residual map; stale legacy entries are dropped."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "all_excluded_v1",
        anchors=(
            _anchor("arm_a", 0.1, in_domain_context="excluded_X"),
            _anchor("arm_b", 0.2, in_domain_context="excluded_Y"),
            _anchor("arm_c", 0.3, in_domain_context="excluded_X"),
        ),
        domain_of_validity={"excluded_contexts": ["excluded_X", "excluded_Y"]},
        residual_map={"arm_a": 0.1, "arm_b": 0.2, "arm_c": 0.3},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "all_excluded_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("all_excluded_v1", path=path)
    assert after is not None
    # All 3 anchors excluded; map empty (every legacy axis dropped).
    assert after.predicted_vs_empirical_residual == {}


def test_refit_eligible_path_still_runs_with_no_nan_no_excluded(tmp_path: Path) -> None:
    """Pre-existing canonical 3+ anchor refit path still fires correctly when
    NO excluded_contexts is declared AND no NaN in stored map."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "canonical_3anchor_refit_v1",
        anchors=(
            _anchor("arm_a", 0.0),
            _anchor("arm_b", 0.5),
            _anchor("arm_c", 1.0),
        ),
        residual_map={"stale_alpha_0p15": 0.15},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "canonical_3anchor_refit_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("canonical_3anchor_refit_v1", path=path)
    assert after is not None
    assert after.predicted_vs_empirical_residual == {
        "arm_a": 0.0,
        "arm_b": 0.5,
        "arm_c": 1.0,
    }
