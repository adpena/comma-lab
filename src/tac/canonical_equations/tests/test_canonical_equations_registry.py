# SPDX-License-Identifier: MIT
"""Tests for tac.canonical_equations registry (Catalog #344 anchor)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations import (
    CANONICAL_EQUATION_SCHEMA_VERSION,
    CanonicalEquation,
    EmpiricalAnchor,
    InvalidEquationError,
    RECALIBRATE_ON_NEW_ANCHORS,
    get_equation_by_id,
    load_equation_registry_strict,
    load_registry_events_lenient,
    query_equations,
    query_equations_by_consumer,
    query_equations_by_domain,
    query_equations_by_producer,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
)
from tac.canonical_equations.registry import (
    CanonicalEquationsRegistryCorruptError,
    _equation_from_dict,
    auto_recalibrate_from_continual_learning_posterior,
)
from tac.provenance.builders import build_provenance_for_predicted


def _design_prov():
    return build_provenance_for_predicted(
        model_id="test.fixture.v1",
        inputs_sha256="0" * 64,
    )


def _make_eq(eq_id: str = "test_equation_v1", consumers=("tac.foo",)) -> CanonicalEquation:
    return CanonicalEquation(
        equation_id=eq_id,
        name="Test equation",
        one_line_summary="Test summary",
        latex_form=r"y = mx + b",
        python_callable_module_path="tac.foo:bar",
        domain_of_validity={"axis": "test"},
        units_in={"x": "float"},
        units_out={"y": "float"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-05-19T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=consumers,
        canonical_producers=("tac.bar",),
        provenance=_design_prov(),
    )


def _make_anchor(anchor_id: str = "test_anchor_v1") -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc="2026-05-19T01:00:00Z",
        inputs={"x": 1.0},
        predicted_output={"y": 2.0},
        empirical_output={"y": 2.1},
        residual=0.05,
        source_artifact="experiments/results/test_anchor",
        measurement_method="test_method",
        provenance=_design_prov(),
    )


# ---------------- CanonicalEquation dataclass invariants ----------------


def test_equation_id_must_match_canonical_pattern():
    with pytest.raises(InvalidEquationError, match="snake_case_vN"):
        _make_eq(eq_id="BadID")
    with pytest.raises(InvalidEquationError, match="snake_case_vN"):
        _make_eq(eq_id="no_version_suffix")
    # Accepts canonical
    eq = _make_eq(eq_id="ok_equation_v42")
    assert eq.equation_id == "ok_equation_v42"


def test_equation_summary_length_capped():
    eq = _make_eq()
    with pytest.raises(InvalidEquationError, match="exceeds 200-char limit"):
        CanonicalEquation(
            equation_id="long_summary_v1",
            name="x",
            one_line_summary="x" * 201,
            latex_form="x",
            python_callable_module_path="tac.x",
            domain_of_validity={},
            units_in={},
            units_out={},
            empirical_anchors=(),
            predicted_vs_empirical_residual={},
            last_calibration_utc="2026-05-19T00:00:00Z",
            next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
            canonical_consumers=("tac.foo",),
            canonical_producers=(),
            provenance=_design_prov(),
        )


def test_equation_callable_path_validated():
    with pytest.raises(InvalidEquationError, match="dotted-module-path"):
        CanonicalEquation(
            equation_id="bad_callable_v1",
            name="x",
            one_line_summary="x",
            latex_form="x",
            python_callable_module_path="bad-dashes-not-snake",
            domain_of_validity={},
            units_in={},
            units_out={},
            empirical_anchors=(),
            predicted_vs_empirical_residual={},
            last_calibration_utc="2026-05-19T00:00:00Z",
            next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
            canonical_consumers=("tac.foo",),
            canonical_producers=(),
            provenance=_design_prov(),
        )


def test_orphan_equation_rejected():
    """Per operator NON-NEGOTIABLE: no producers AND no consumers = orphan."""
    with pytest.raises(InvalidEquationError, match="orphan equations are forbidden"):
        CanonicalEquation(
            equation_id="orphan_v1",
            name="x",
            one_line_summary="x",
            latex_form="x",
            python_callable_module_path="tac.foo",
            domain_of_validity={},
            units_in={},
            units_out={},
            empirical_anchors=(),
            predicted_vs_empirical_residual={},
            last_calibration_utc="2026-05-19T00:00:00Z",
            next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
            canonical_consumers=(),
            canonical_producers=(),
            provenance=_design_prov(),
        )


def test_recalibration_trigger_validated():
    with pytest.raises(InvalidEquationError, match="next_recalibration_trigger"):
        CanonicalEquation(
            equation_id="bad_trigger_v1",
            name="x",
            one_line_summary="x",
            latex_form="x",
            python_callable_module_path="tac.foo",
            domain_of_validity={},
            units_in={},
            units_out={},
            empirical_anchors=(),
            predicted_vs_empirical_residual={},
            last_calibration_utc="2026-05-19T00:00:00Z",
            next_recalibration_trigger="bogus_trigger",
            canonical_consumers=("tac.foo",),
            canonical_producers=(),
            provenance=_design_prov(),
        )


def test_residuals_must_be_non_negative():
    with pytest.raises(InvalidEquationError, match="must be >= 0"):
        CanonicalEquation(
            equation_id="bad_residual_v1",
            name="x",
            one_line_summary="x",
            latex_form="x",
            python_callable_module_path="tac.foo",
            domain_of_validity={},
            units_in={},
            units_out={},
            empirical_anchors=(),
            predicted_vs_empirical_residual={"axis": -1.0},
            last_calibration_utc="2026-05-19T00:00:00Z",
            next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
            canonical_consumers=("tac.foo",),
            canonical_producers=(),
            provenance=_design_prov(),
        )


def test_is_well_calibrated_empty_anchors_false():
    eq = _make_eq()
    assert not eq.is_well_calibrated


def test_is_well_calibrated_threshold():
    eq = _make_eq()
    anchor = _make_anchor()
    updated = eq.with_new_anchor(anchor)
    assert updated.is_well_calibrated  # residual=0.05 < 2.0
    # Add anchor with residual > 2.0
    big = EmpiricalAnchor(
        anchor_id="big_v1",
        measurement_utc="2026-05-19T02:00:00Z",
        inputs={},
        predicted_output=1,
        empirical_output=100,
        residual=99.0,
        source_artifact="x",
        measurement_method="big_axis",
        provenance=_design_prov(),
    )
    updated2 = updated.with_new_anchor(big)
    assert not updated2.is_well_calibrated


def test_to_dict_round_trip():
    eq = _make_eq()
    d = eq.to_dict()
    assert d["schema_version"] == CANONICAL_EQUATION_SCHEMA_VERSION
    assert d["equation_id"] == eq.equation_id
    assert isinstance(d["provenance"], dict)
    recon = _equation_from_dict(d)
    assert recon.equation_id == eq.equation_id


# ---------------- EmpiricalAnchor invariants ----------------


def test_anchor_residual_nan_rejected():
    import math

    with pytest.raises(InvalidEquationError, match="must not be NaN"):
        EmpiricalAnchor(
            anchor_id="nan_v1",
            measurement_utc="2026-05-19T00:00:00Z",
            inputs={},
            predicted_output=0,
            empirical_output=0,
            residual=float("nan"),
            source_artifact="x",
            measurement_method="x",
            provenance=_design_prov(),
        )


def test_anchor_iso_utc_required():
    with pytest.raises(InvalidEquationError, match="not valid ISO"):
        EmpiricalAnchor(
            anchor_id="bad_utc_v1",
            measurement_utc="yesterday",
            inputs={},
            predicted_output=0,
            empirical_output=0,
            residual=0.0,
            source_artifact="x",
            measurement_method="x",
            provenance=_design_prov(),
        )


# ---------------- Registry persistence ----------------


def test_register_and_query_roundtrip(tmp_path: Path):
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    eq = _make_eq(eq_id="round_trip_v1", consumers=("tac.example_consumer",))
    register_canonical_equation(eq, path=path, lock_path=lock)
    loaded = query_equations(path=path)
    assert len(loaded) == 1
    assert loaded[0].equation_id == "round_trip_v1"


def test_register_then_update_with_anchor(tmp_path: Path):
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    eq = _make_eq(eq_id="anchor_update_v1")
    register_canonical_equation(eq, path=path, lock_path=lock)
    anchor = _make_anchor("anchor_a")
    updated = update_equation_with_empirical_anchor(
        "anchor_update_v1", anchor, path=path, lock_path=lock
    )
    assert len(updated.empirical_anchors) == 1
    # Re-query; latest payload wins.
    re_loaded = get_equation_by_id("anchor_update_v1", path=path)
    assert re_loaded is not None
    assert len(re_loaded.empirical_anchors) == 1


def test_update_unknown_equation_raises(tmp_path: Path):
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    anchor = _make_anchor("unknown_anchor")
    with pytest.raises(InvalidEquationError, match="not found in registry"):
        update_equation_with_empirical_anchor(
            "nonexistent_v1", anchor, path=path, lock_path=lock
        )


def test_load_strict_raises_on_corrupt_json(tmp_path: Path):
    path = tmp_path / "corrupt.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(CanonicalEquationsRegistryCorruptError):
        load_equation_registry_strict(path)


def test_load_lenient_skips_corrupt_lines(tmp_path: Path):
    path = tmp_path / "mixed.jsonl"
    path.write_text("not json\n" + json.dumps({"ok": 1}) + "\n", encoding="utf-8")
    rows = load_registry_events_lenient(path)
    assert len(rows) == 1


def test_query_returns_empty_when_no_path():
    # Use a path that doesn't exist.
    out = query_equations(path=Path("/nonexistent/registry.jsonl"))
    assert out == []


# ---------------- Query helpers ----------------


def test_query_by_consumer(tmp_path: Path):
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    eq1 = _make_eq("query_consumer_a_v1", consumers=("tac.mod_a",))
    eq2 = _make_eq("query_consumer_b_v1", consumers=("tac.mod_b",))
    register_canonical_equation(eq1, path=path, lock_path=lock)
    register_canonical_equation(eq2, path=path, lock_path=lock)
    out = query_equations_by_consumer("tac.mod_a", path=path)
    ids = [e.equation_id for e in out]
    assert "query_consumer_a_v1" in ids
    assert "query_consumer_b_v1" not in ids


def test_query_by_producer(tmp_path: Path):
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    eq = _make_eq("query_producer_v1")
    register_canonical_equation(eq, path=path, lock_path=lock)
    out = query_equations_by_producer("tac.bar", path=path)
    assert len(out) == 1


def test_query_by_domain(tmp_path: Path):
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    eq = _make_eq("query_domain_v1")  # domain has axis=test
    register_canonical_equation(eq, path=path, lock_path=lock)
    out = query_equations_by_domain("test", path=path)
    assert len(out) == 1
    out_empty = query_equations_by_domain("nonexistent", path=path)
    assert out_empty == []


def test_get_equation_by_id_missing_returns_none(tmp_path: Path):
    out = get_equation_by_id("never_existed_v1", path=tmp_path / "empty.jsonl")
    assert out is None


# ---------------- Latest-row-wins semantics ----------------


def test_latest_payload_wins(tmp_path: Path):
    """Re-registering an equation appends a new event; query returns latest."""
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    eq1 = _make_eq("latest_wins_v1")
    register_canonical_equation(eq1, path=path, lock_path=lock)
    # Re-register with anchor added.
    eq2 = eq1.with_new_anchor(_make_anchor("latest_anchor_v1"))
    register_canonical_equation(eq2, path=path, lock_path=lock)
    out = get_equation_by_id("latest_wins_v1", path=path)
    assert out is not None
    assert len(out.empirical_anchors) == 1


def test_recalibration_report_summary(tmp_path: Path):
    path = tmp_path / "test_registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    eq = _make_eq("recal_report_v1")
    register_canonical_equation(eq, path=path, lock_path=lock)
    report = auto_recalibrate_from_continual_learning_posterior(path=path)
    assert report.equations_checked == 1
    assert report.new_anchors_absorbed == 0
    assert "recal_report_v1" in report.per_equation_summary


# ---------------- Atomic write / re-entry semantics ----------------


def test_save_outside_lock_raises():
    """_save_ledger must refuse calls outside the lock per Catalog #140."""
    from tac.canonical_equations.registry import _save_ledger

    with pytest.raises(RuntimeError, match="WITHOUT holding _registry_lock"):
        _save_ledger([], path=Path("/tmp/never_written.jsonl"))


def test_with_new_anchor_immutability():
    eq = _make_eq()
    anchor = _make_anchor()
    updated = eq.with_new_anchor(anchor)
    # Original unchanged.
    assert len(eq.empirical_anchors) == 0
    assert len(updated.empirical_anchors) == 1
    # last_calibration_utc bumped.
    assert updated.last_calibration_utc >= eq.last_calibration_utc
