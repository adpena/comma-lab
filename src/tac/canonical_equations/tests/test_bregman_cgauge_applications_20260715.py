from __future__ import annotations

import importlib

import numpy as np
import pytest

from tac.canonical_equations.bregman_v9_surfaces_20260714 import (
    APPLICATION_EQUATION_IDS,
    CGUAGE_HESSIAN_EQUATION_ID,
    CLOSED_FORM_EQUATION_ID,
    NONNEGATIVITY_EQUATION_ID,
    RIGHT_CENTROID_EQUATION_ID,
    SIGMA_PROPAGATION_EQUATION_ID,
    build_bregman_cgauge_application_equations_v1,
    populate_bregman_cgauge_application_equations_v1,
)
from tac.canonical_equations.registry import (
    get_equation_by_id,
    load_registry_events_lenient,
    query_equations,
)
from tac.information_geometry.bregman_v9_surfaces import (
    GeometryValidationError,
    affine_legendre_logsumexp_summary,
    bregman_divergence,
    categorical_bregman_sigma_propagation,
    categorical_kl_from_logits,
    categorical_left_data_centroid,
    categorical_log_partition_hessian,
    categorical_negative_entropy_bregman,
    categorical_right_data_centroid,
    categorical_softmax,
    exponential_family_sigma_kl_error,
    logsumexp_bregman,
    logsumexp_bregman_closed_form_summary,
    positive_unscented_sigma_points,
    require_nonnegative_bregman,
)


def test_categorical_ground_metric_is_covariance_hessian_on_quotient() -> None:
    logits = np.asarray([0.7, -0.3, 0.2, -0.8], dtype=np.float64)
    probability = categorical_softmax(logits)
    hessian = categorical_log_partition_hessian(logits)

    np.testing.assert_allclose(
        hessian,
        np.diag(probability) - np.outer(probability, probability),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(hessian, hessian.T, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(hessian @ np.ones(logits.size), 0.0, atol=1.0e-16)
    assert np.linalg.eigvalsh(hessian).min() >= -1.0e-15


def test_logsumexp_bregman_is_reverse_categorical_kl_and_gauge_invariant() -> None:
    point = np.asarray([0.9, -0.4, 0.1], dtype=np.float64)
    reference = np.asarray([-0.2, 0.6, -0.3], dtype=np.float64)
    p_point = categorical_softmax(point)
    p_reference = categorical_softmax(reference)
    expected = categorical_negative_entropy_bregman(p_reference, p_point)

    assert logsumexp_bregman(point, reference) == pytest.approx(
        expected, abs=2.0e-16
    )
    assert logsumexp_bregman(point + 17.0, reference - 8.0) == pytest.approx(
        expected, abs=4.0e-15
    )
    assert logsumexp_bregman(point + 5.0, point) == pytest.approx(0.0, abs=1.0e-15)


def test_dual_reversal_and_gradient_pairing_cancel_exactly_without_solve() -> None:
    summary = logsumexp_bregman_closed_form_summary(
        np.asarray([0.8, -0.3, 0.2]),
        np.asarray([-0.1, 0.6, -0.4]),
    )
    assert summary["dual_identity_abs_error"] <= 1.0e-15
    assert summary["cancellation_abs_error"] <= 1.0e-15
    assert "raw_dual_euclidean" not in summary
    assert "fisher_natural" not in summary


def test_dual_closed_form_stays_finite_for_extreme_finite_logits() -> None:
    point = np.asarray([1000.0, -1000.0, 0.0], dtype=np.float64)
    reference = np.asarray([-1000.0, 1000.0, 0.0], dtype=np.float64)
    summary = logsumexp_bregman_closed_form_summary(point, reference)

    assert np.isfinite(summary["primal_bregman"])
    assert np.isfinite(summary["reversed_dual_bregman"])
    assert summary["dual_identity_abs_error"] <= 1.0e-12
    assert categorical_kl_from_logits(reference, point) == pytest.approx(
        summary["primal_bregman"], abs=1.0e-12
    )


def test_affine_legendre_covariance_cancels_linear_and_constant_gauge() -> None:
    kwargs = {
        "point": np.asarray([0.5, -0.2]),
        "reference": np.asarray([-0.4, 0.7]),
        "matrix": np.asarray([[1.0, 0.2], [-0.3, 1.1], [0.4, 0.6]]),
        "offset": np.asarray([0.1, -0.2, 0.3]),
        "scale": 1.6,
        "linear_term": np.asarray([0.7, -1.3]),
        "constant": 9.0,
    }
    summary = affine_legendre_logsumexp_summary(**kwargs)
    altered_gauge = affine_legendre_logsumexp_summary(
        **{**kwargs, "linear_term": np.asarray([-4.0, 2.5]), "constant": -31.0}
    )

    assert summary["covariance_abs_error"] <= 2.0e-15
    assert altered_gauge["covariance_abs_error"] <= 4.0e-15
    assert summary["gauged_bregman"] == pytest.approx(
        altered_gauge["gauged_bregman"], abs=5.0e-15
    )


def test_negative_divergence_guard_refuses_nonconvex_generator() -> None:
    def concave(x):
        return -0.5 * float(x @ x)

    def concave_gradient(x):
        return -x

    value = bregman_divergence(
        concave,
        concave_gradient,
        np.asarray([1.0, -1.0]),
        np.asarray([0.0, 0.0]),
    )
    assert value < 0.0
    with pytest.raises(GeometryValidationError, match="negative beyond tolerance"):
        require_nonnegative_bregman(value)
    assert require_nonnegative_bregman(-5.0e-13, atol=1.0e-12) == 0.0


def test_right_data_centroid_is_dual_mean_and_opposite_orientation_is_explicit() -> None:
    samples = np.asarray(
        [[0.4, -0.2, 0.1], [-0.3, 0.7, -0.1], [0.2, 0.0, -0.5]],
        dtype=np.float64,
    )
    weights = np.asarray([0.2, 0.5, 0.3], dtype=np.float64)
    right = categorical_right_data_centroid(samples, weights)
    left = categorical_left_data_centroid(samples, weights)

    np.testing.assert_allclose(
        right["recovered_probability"],
        right["dual_probability_mean"],
        rtol=0.0,
        atol=2.0e-16,
    )
    assert right["first_order_residual_linf"] <= 2.0e-16
    np.testing.assert_allclose(left, weights @ samples - (weights @ samples).mean())
    assert not np.allclose(left, right["centroid_logits_zero_mean"])

    center = right["centroid_logits_zero_mean"]
    objective = right["weighted_objective"]
    for perturbation in (
        np.asarray([1.0e-4, -1.0e-4, 0.0]),
        np.asarray([0.0, 1.0e-4, -1.0e-4]),
    ):
        perturbed = sum(
            weight * logsumexp_bregman(center + perturbation, sample)
            for weight, sample in zip(weights, samples, strict=True)
        )
        assert perturbed >= objective - 1.0e-15


def test_positive_sigma_rule_exactly_matches_input_moments() -> None:
    mean = np.asarray([0.3, -0.4, 0.2], dtype=np.float64)
    covariance = np.asarray(
        [[0.4, 0.03, -0.02], [0.03, 0.25, 0.01], [-0.02, 0.01, 0.3]],
        dtype=np.float64,
    )
    points, weights = positive_unscented_sigma_points(mean, covariance, kappa=0.7)
    reconstructed_mean = weights @ points
    centered = points - reconstructed_mean
    reconstructed_covariance = np.einsum(
        "n,ni,nj->ij", weights, centered, centered
    )

    assert np.all(weights > 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=3.0e-16)
    np.testing.assert_allclose(reconstructed_mean, mean, rtol=0.0, atol=1.0e-16)
    np.testing.assert_allclose(
        reconstructed_covariance, covariance, rtol=0.0, atol=2.0e-16
    )
    with pytest.raises(GeometryValidationError, match="kappa"):
        positive_unscented_sigma_points(mean, covariance, kappa=0.0)


def test_sigma_propagation_labels_nonlinear_output_and_ef_exactness_condition() -> None:
    summary = categorical_bregman_sigma_propagation(
        np.asarray([0.2, -0.1]),
        np.asarray([[0.3, 0.04], [0.04, 0.2]]),
        lambda x: np.asarray([x[0], x[1], x[0] * x[1]]),
        kappa=1.0,
    )
    assert summary["input_mean_abs_error"] <= 1.0e-16
    assert summary["input_covariance_abs_error"] <= 2.0e-16
    assert "NONLINEAR_OUTPUT_APPROXIMATE" in summary["quadrature_scope"]
    assert summary["exact_bregman_dispersion"] >= 0.0
    assert summary["local_hessian_dispersion"] >= 0.0
    assert exponential_family_sigma_kl_error(
        np.asarray([0.3, -0.2]),
        np.asarray([-0.1, 0.4]),
        np.asarray([0.6, 0.4]),
        np.asarray([0.6, 0.4]),
    ) == pytest.approx(0.0)
    assert exponential_family_sigma_kl_error(
        np.asarray([0.3, -0.2]),
        np.asarray([-0.1, 0.4]),
        np.asarray([0.6, 0.4]),
        np.asarray([0.61, 0.39]),
    ) != pytest.approx(0.0)


def test_application_equations_encode_scope_and_no_fake_levers() -> None:
    equations = build_bregman_cgauge_application_equations_v1()
    assert tuple(equation.equation_id for equation in equations) == (
        APPLICATION_EQUATION_IDS
    )
    by_id = {equation.equation_id: equation for equation in equations}
    assert by_id[CGUAGE_HESSIAN_EQUATION_ID].domain_of_validity[
        "affine_legendre_model_status"
    ] == "IMPLEMENTATION_CUSTODY_GAP_ONLY"
    assert by_id[CLOSED_FORM_EQUATION_ID].domain_of_validity[
        "raw_dual_no_solve"
    ].endswith("squared Hessian only")
    assert by_id[NONNEGATIVITY_EQUATION_ID].canonical_consumers == ()
    assert by_id[RIGHT_CENTROID_EQUATION_ID].canonical_consumers == ()
    assert by_id[SIGMA_PROPAGATION_EQUATION_ID].canonical_consumers == ()
    for equation in equations:
        assert equation.domain_of_validity["dsl_wire_status"].startswith("OWED")
        assert equation.domain_of_validity["score_claim"] is False
        assert equation.domain_of_validity["promotion_eligible"] is False


def test_application_population_is_explicit_and_uses_registry_api(tmp_path) -> None:
    registry_path = tmp_path / "canonical_equations_registry.jsonl"
    lock_path = tmp_path / "canonical_equations_registry.jsonl.lock"
    module = importlib.import_module(
        "tac.canonical_equations.bregman_v9_surfaces_20260714"
    )
    assert module.APPLICATION_EQUATION_IDS == APPLICATION_EQUATION_IDS
    assert not registry_path.exists(), "import must not mutate a registry"

    equations = populate_bregman_cgauge_application_equations_v1(
        path=registry_path,
        lock_path=lock_path,
        agent="codex",
        subagent_id="bregman_all_surfaces_504_test",
    )
    assert tuple(equation.equation_id for equation in equations) == (
        APPLICATION_EQUATION_IDS
    )
    assert [event["equation_id"] for event in load_registry_events_lenient(registry_path)] == (
        list(APPLICATION_EQUATION_IDS)
    )
    assert [equation.equation_id for equation in query_equations(path=registry_path)] == (
        list(APPLICATION_EQUATION_IDS)
    )
    for equation_id in APPLICATION_EQUATION_IDS:
        assert get_equation_by_id(equation_id, path=registry_path) is not None


def test_live_registry_surfaces_task_504_application_equations_once() -> None:
    ids = [equation.equation_id for equation in query_equations()]
    for equation_id in APPLICATION_EQUATION_IDS:
        assert ids.count(equation_id) == 1
