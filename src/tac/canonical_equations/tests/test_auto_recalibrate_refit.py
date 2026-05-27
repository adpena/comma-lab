# SPDX-License-Identifier: MIT
"""Tests for the auto_recalibrate_from_continual_learning_posterior refit fix.

Per STUB-AUDIT-AND-FIX wave 2026-05-27 (operator directive "fix all stubs and
continue iterating and optimizing and auditing"). The canonical orphan-stub:
``auto_recalibrate_from_continual_learning_posterior`` previously no-op'd
(``equations_recalibrated=0``) even when its OWN documented trigger condition
(``when_3+_new_empirical_anchors_in_domain``) was satisfied. The fix re-derives
``predicted_vs_empirical_residual`` from the equation's own landed anchors when
the 3+ trigger fires, appends a canonical EVENT_RECALIBRATED row, and bumps
``last_calibration_utc`` — evidence-faithful (no synthesized anchors per
Catalog #287/#323).
"""
from __future__ import annotations

import math
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    RECALIBRATE_ON_RESIDUAL_DRIFT,
    RECALIBRATE_ON_PARAMETER_REFIT,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.registry import (
    EVENT_RECALIBRATED,
    auto_recalibrate_from_continual_learning_posterior,
    get_equation_by_id,
    load_registry_events_lenient,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_predicted


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _prov():
    return build_provenance_for_predicted(
        model_id="test.auto_recalibrate.v1",
        inputs_sha256="0" * 64,
    )


def _anchor(method: str, residual: float) -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=f"anchor_{method}",
        measurement_utc="2026-05-27T00:00:00Z",
        inputs={"x": 1},
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
    trigger: str = RECALIBRATE_ON_NEW_ANCHORS,
) -> CanonicalEquation:
    return CanonicalEquation(
        equation_id=eq_id,
        name=eq_id.replace("_", " "),
        one_line_summary="test equation for auto-recalibrate refit",
        latex_form=r"\Delta S = \alpha",
        python_callable_module_path="tac.canonical_equations.tests.fixture:predict",
        domain_of_validity={"test": True},
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
# Core refit behavior — the fix
# ---------------------------------------------------------------------------


def test_refit_fires_when_3plus_anchors_and_stored_map_stale(tmp_path: Path) -> None:
    """3+ anchors + stored map disagreeing with anchors -> refit fires."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "stale_prior_v1",
        anchors=(
            _anchor("arm_a", 0.0),
            _anchor("arm_b", 0.0277),
            _anchor("arm_c", 0.5116),
        ),
        # Stale map: claims a 0.15 prior axis the anchors no longer support.
        residual_map={"closed_form_alpha_0p15_prior": 0.15},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "stale_prior_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("stale_prior_v1", path=path)
    assert after is not None
    # Refit map == per-anchor-method residuals; stale prior axis dropped.
    assert dict(after.predicted_vs_empirical_residual) == {
        "arm_a": 0.0,
        "arm_b": 0.0277,
        "arm_c": 0.5116,
    }
    # last_calibration_utc bumped off the 2026-05-01 placeholder.
    assert after.last_calibration_utc != "2026-05-01T00:00:00Z"


def test_refit_emits_event_recalibrated_row(tmp_path: Path) -> None:
    """The refit appends a canonical EVENT_RECALIBRATED ledger row."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "emits_event_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"stale": 9.9},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(
        "emits_event_v1", path=path, lock_path=lock
    )
    rows = load_registry_events_lenient(path)
    event_types = [r.get("event_type") for r in rows if r.get("equation_id") == "emits_event_v1"]
    assert EVENT_RECALIBRATED in event_types


def test_refit_appends_only_preserves_prior_payload(tmp_path: Path) -> None:
    """APPEND-ONLY: the original 'registered' row is preserved verbatim."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "append_only_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"stale": 1.0},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(
        "append_only_v1", path=path, lock_path=lock
    )
    rows = [r for r in load_registry_events_lenient(path) if r.get("equation_id") == "append_only_v1"]
    # First row is the original 'registered' event with the stale map intact.
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_payload"]["predicted_vs_empirical_residual"] == {"stale": 1.0}


def test_refit_idempotent_second_run_no_op(tmp_path: Path) -> None:
    """Running twice: second run recalibrates 0 (no drift remaining)."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "idempotent_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"stale": 1.0},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep1 = auto_recalibrate_from_continual_learning_posterior(
        "idempotent_v1", path=path, lock_path=lock
    )
    assert rep1.equations_recalibrated == 1
    rep2 = auto_recalibrate_from_continual_learning_posterior(
        "idempotent_v1", path=path, lock_path=lock
    )
    assert rep2.equations_recalibrated == 0


def test_refit_no_write_when_map_already_matches(tmp_path: Path) -> None:
    """If stored map already equals anchor-derived map, no event is appended."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "already_matches_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"a": 0.1, "b": 0.2, "c": 0.3},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    before_rows = len(load_registry_events_lenient(path))
    rep = auto_recalibrate_from_continual_learning_posterior(
        "already_matches_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 0
    assert len(load_registry_events_lenient(path)) == before_rows


def test_refit_drops_stale_nan_residual_sentinel(tmp_path: Path) -> None:
    """A stored NaN-sentinel residual is dropped by the anchor-derived refit."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "nan_sentinel_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"pending_axis": math.nan, "a": 0.1, "b": 0.2, "c": 0.3},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "nan_sentinel_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 1
    after = get_equation_by_id("nan_sentinel_v1", path=path)
    assert "pending_axis" not in after.predicted_vs_empirical_residual
    assert not any(v != v for v in after.predicted_vs_empirical_residual.values())


# ---------------------------------------------------------------------------
# Trigger discipline — refit ONLY fires for the right trigger + count
# ---------------------------------------------------------------------------


def test_no_refit_below_3_anchors(tmp_path: Path) -> None:
    """2 anchors -> trigger NOT satisfied -> no refit (legitimate deferral)."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "two_anchors_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2)),
        residual_map={"stale": 9.0},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "two_anchors_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 0
    after = get_equation_by_id("two_anchors_v1", path=path)
    # Stored stale map untouched (awaiting 3rd anchor).
    assert dict(after.predicted_vs_empirical_residual) == {"stale": 9.0}


def test_no_refit_for_residual_drift_trigger(tmp_path: Path) -> None:
    """when_residual_drift_exceeds_2x equations are NOT auto-refit here."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "drift_trigger_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"stale": 9.0},
        trigger=RECALIBRATE_ON_RESIDUAL_DRIFT,
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "drift_trigger_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 0


def test_no_refit_for_operator_only_trigger(tmp_path: Path) -> None:
    """when_operator_invokes equations are NOT auto-refit by this helper."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "operator_only_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"stale": 9.0},
        trigger=RECALIBRATE_ON_PARAMETER_REFIT,
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "operator_only_v1", path=path, lock_path=lock
    )
    assert rep.equations_recalibrated == 0


def test_zero_anchor_equation_reports_but_not_refit(tmp_path: Path) -> None:
    """Design-only equation (0 anchors) is reported but never refit."""
    path, lock = _setup(tmp_path)
    eq = _eq("design_only_v1")
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "design_only_v1", path=path, lock_path=lock
    )
    assert rep.equations_checked == 1
    assert rep.equations_recalibrated == 0
    assert rep.new_anchors_absorbed == 0
    assert "design_only_v1" in rep.per_equation_summary


# ---------------------------------------------------------------------------
# Report shape + summary fields
# ---------------------------------------------------------------------------


def test_report_summary_carries_refit_flags(tmp_path: Path) -> None:
    """Per-equation summary carries refit_eligible + recalibrated flags."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "summary_flags_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"stale": 1.0},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "summary_flags_v1", path=path, lock_path=lock
    )
    s = rep.per_equation_summary["summary_flags_v1"]
    assert s["refit_eligible"] is True
    assert s["recalibrated"] is True
    assert s["anchor_count"] == 3


def test_absorbed_count_equals_changed_axes(tmp_path: Path) -> None:
    """new_anchors_absorbed counts changed/added/removed summary axes."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "absorbed_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        # stored has 1 stale key; refit adds 3 new keys, drops 1 -> 4 changed.
        residual_map={"stale": 9.0},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    rep = auto_recalibrate_from_continual_learning_posterior(
        "absorbed_v1", path=path, lock_path=lock
    )
    assert rep.new_anchors_absorbed == 4


def test_full_registry_scan_recalibrates_only_eligible(tmp_path: Path) -> None:
    """Full scan (no equation_id) refits only the eligible equations."""
    path, lock = _setup(tmp_path)
    register_canonical_equation(
        _eq(
            "eligible_v1",
            anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
            residual_map={"stale": 1.0},
        ),
        path=path,
        lock_path=lock,
    )
    register_canonical_equation(
        _eq("ineligible_v1", anchors=(_anchor("a", 0.1),), residual_map={"a": 0.1}),
        path=path,
        lock_path=lock,
    )
    rep = auto_recalibrate_from_continual_learning_posterior(path=path, lock_path=lock)
    assert rep.equations_checked == 2
    assert rep.equations_recalibrated == 1
    assert rep.per_equation_summary["eligible_v1"]["recalibrated"] is True
    assert rep.per_equation_summary["ineligible_v1"]["recalibrated"] is False


def test_no_synthesized_anchors_count_preserved(tmp_path: Path) -> None:
    """Refit never synthesizes anchors (Catalog #287); anchor count unchanged."""
    path, lock = _setup(tmp_path)
    eq = _eq(
        "no_synth_v1",
        anchors=(_anchor("a", 0.1), _anchor("b", 0.2), _anchor("c", 0.3)),
        residual_map={"stale": 1.0},
    )
    register_canonical_equation(eq, path=path, lock_path=lock)
    auto_recalibrate_from_continual_learning_posterior(
        "no_synth_v1", path=path, lock_path=lock
    )
    after = get_equation_by_id("no_synth_v1", path=path)
    assert len(after.empirical_anchors) == 3


def test_refit_after_real_anchor_append_path(tmp_path: Path) -> None:
    """End-to-end: append 3 anchors via canonical path then recalibrate refreshes."""
    path, lock = _setup(tmp_path)
    eq = _eq("e2e_v1", residual_map={})
    register_canonical_equation(eq, path=path, lock_path=lock)
    for m, r in (("a", 0.1), ("b", 0.2), ("c", 0.9)):
        update_equation_with_empirical_anchor(
            "e2e_v1", _anchor(m, r), path=path, lock_path=lock
        )
    # with_new_anchor already keeps the map current, so refit should be a no-op
    # here — proving the gate-fix is consistent with the append path.
    rep = auto_recalibrate_from_continual_learning_posterior(
        "e2e_v1", path=path, lock_path=lock
    )
    after = get_equation_by_id("e2e_v1", path=path)
    assert dict(after.predicted_vs_empirical_residual) == {"a": 0.1, "b": 0.2, "c": 0.9}
    assert rep.equations_recalibrated == 0
