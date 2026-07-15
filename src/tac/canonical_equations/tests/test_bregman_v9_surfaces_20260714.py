from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.bregman_v9_surfaces_20260714 import (
    AXIS,
    BINDING_ARTIFACT,
    EQUATION_ID,
    INFORMATION_GEOMETRY_HELPER,
    MAX_PRIMAL_EXACT_DUAL_ERROR,
    MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR,
    RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES,
    REAL_N600_SELECTION_STATUS,
    STATE_COUNT,
    build_bregman_dual_metric_squared_hessian_v1,
    populate_bregman_dual_metric_squared_hessian_v1,
)
from tac.canonical_equations.registry import (
    get_equation_by_id,
    load_registry_events_lenient,
)
from tac.information_geometry.bregman_v9_surfaces import (
    DELTA_ETA_CONSISTENCY_ATOL,
    GeometryValidationError,
    fisher_natural_cotangent_quadratic,
    local_hessian_dual_geometry_summary,
    primal_hessian_quadratic,
    raw_dual_euclidean_quadratic,
    squared_hessian_quadratic,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hessian = np.array([[3.0, 0.75], [0.75, 1.5]], dtype=np.float64)
    delta_theta = np.array([1.25, -0.5], dtype=np.float64)
    delta_eta = hessian @ delta_theta
    return hessian, delta_theta, delta_eta


def test_four_quantities_preserve_the_two_correct_identities() -> None:
    hessian, delta_theta, delta_eta = _fixture()
    summary = local_hessian_dual_geometry_summary(hessian, delta_theta, delta_eta)

    assert summary["primal_hessian"] == pytest.approx(
        primal_hessian_quadratic(hessian, delta_theta)
    )
    assert summary["fisher_natural_cotangent"] == pytest.approx(
        fisher_natural_cotangent_quadratic(hessian, delta_eta)
    )
    assert summary["raw_dual_euclidean"] == pytest.approx(
        raw_dual_euclidean_quadratic(delta_eta)
    )
    assert summary["squared_hessian"] == pytest.approx(
        squared_hessian_quadratic(hessian, delta_theta)
    )
    assert summary["primal_hessian"] == pytest.approx(
        summary["fisher_natural_cotangent"], abs=1.0e-13
    )
    assert summary["raw_dual_euclidean"] == pytest.approx(
        summary["squared_hessian"], abs=1.0e-13
    )
    assert not np.isclose(summary["raw_dual_euclidean"], summary["primal_hessian"])


def test_identity_bearing_aggregate_rejects_inconsistent_dual_coordinate() -> None:
    hessian, delta_theta, delta_eta = _fixture()
    inconsistent_delta_eta = delta_eta.copy()
    inconsistent_delta_eta[0] += 10.0 * DELTA_ETA_CONSISTENCY_ATOL

    with pytest.raises(
        GeometryValidationError,
        match=r"delta_eta must equal hessian @ delta_theta within fp64 tolerance",
    ):
        local_hessian_dual_geometry_summary(
            hessian,
            delta_theta,
            inconsistent_delta_eta,
        )


def test_identity_bearing_aggregate_accepts_fp64_tolerance_roundoff() -> None:
    hessian, delta_theta, delta_eta = _fixture()
    rounded_delta_eta = delta_eta.copy()
    rounded_delta_eta[0] += 0.5 * DELTA_ETA_CONSISTENCY_ATOL

    summary = local_hessian_dual_geometry_summary(
        hessian,
        delta_theta,
        rounded_delta_eta,
    )

    assert summary["primal_hessian"] == pytest.approx(
        summary["fisher_natural_cotangent"], abs=2.0e-12
    )


def test_fisher_natural_path_calls_typed_linear_solve(monkeypatch) -> None:
    hessian, _, delta_eta = _fixture()
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    real_solve = np.linalg.solve

    def recording_solve(matrix, vector):
        calls.append((matrix.copy(), vector.copy()))
        return real_solve(matrix, vector)

    monkeypatch.setattr(np.linalg, "solve", recording_solve)
    monkeypatch.setattr(
        np.linalg,
        "inv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("explicit inverse used")),
    )
    fisher_natural_cotangent_quadratic(hessian, delta_eta)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], hessian)
    np.testing.assert_array_equal(calls[0][1], delta_eta)


@pytest.mark.parametrize(
    ("hessian", "delta_theta", "delta_eta", "message"),
    [
        (np.ones((2, 3)), np.ones(2), np.ones(2), "square"),
        (np.eye(2), np.ones(3), np.ones(2), "delta_theta"),
        (np.eye(2), np.ones(2), np.ones(3), "delta_eta"),
        (np.array([[1.0, 0.5], [0.0, 1.0]]), np.ones(2), np.ones(2), "symmetric"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), np.ones(2), np.ones(2), "positive definite"),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), np.ones(2), np.ones(2), "positive definite"),
        (np.array([[1.0, 0.0], [0.0, np.inf]]), np.ones(2), np.ones(2), "finite"),
    ],
)
def test_invalid_or_non_spd_inputs_fail_closed(
    hessian, delta_theta, delta_eta, message
) -> None:
    with pytest.raises(GeometryValidationError, match=message):
        local_hessian_dual_geometry_summary(hessian, delta_theta, delta_eta)


def test_canonical_equation_anchor_keeps_exact_custody_and_nonpromotion() -> None:
    equation = build_bregman_dual_metric_squared_hessian_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.inputs["state_count"] == STATE_COUNT
    assert anchor.inputs["axis"] == AXIS
    assert anchor.inputs["real_n600_selection_status"] == REAL_N600_SELECTION_STATUS
    assert anchor.empirical_output["raw_dual_vs_ordinary_hessian_mismatches"] == (
        RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES
    )
    assert anchor.empirical_output["max_primal_exact_dual_error"] == (
        MAX_PRIMAL_EXACT_DUAL_ERROR
    )
    assert anchor.empirical_output["max_raw_dual_squared_hessian_error"] == (
        MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR
    )
    assert anchor.empirical_output["score_claim"] is False
    assert anchor.empirical_output["promotion_eligible"] is False
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert equation.provenance.measurement_axis == AXIS
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False
    assert equation.canonical_producers == (INFORMATION_GEOMETRY_HELPER,)
    assert equation.domain_of_validity["binding_artifact"] == BINDING_ARTIFACT
    assert Path(BINDING_ARTIFACT).is_file()


def test_population_is_explicit_and_supports_a_temporary_registry(tmp_path) -> None:
    registry_path = tmp_path / "canonical_equations_registry.jsonl"
    lock_path = tmp_path / "canonical_equations_registry.jsonl.lock"
    module = importlib.import_module(
        "tac.canonical_equations.bregman_v9_surfaces_20260714"
    )
    assert module.EQUATION_ID == EQUATION_ID
    assert not registry_path.exists(), "import must not mutate a registry"

    populated = populate_bregman_dual_metric_squared_hessian_v1(
        path=registry_path,
        lock_path=lock_path,
        agent="codex",
        subagent_id="finding1_test",
    )
    assert populated.equation_id == EQUATION_ID
    reloaded = get_equation_by_id(EQUATION_ID, path=registry_path)
    assert reloaded is not None
    assert reloaded.equation_id == populated.equation_id
    assert reloaded.empirical_anchors[0].empirical_output == (
        populated.empirical_anchors[0].empirical_output
    )
    events = load_registry_events_lenient(registry_path)
    assert [event["equation_id"] for event in events] == [EQUATION_ID]


def test_live_registry_has_no_duplicate_equation_id() -> None:
    matching = [
        event
        for event in load_registry_events_lenient()
        if event.get("equation_id") == EQUATION_ID
    ]
    assert len(matching) <= 1
