# SPDX-License-Identifier: MIT
"""Canonical law separating Fisher-natural and raw-dual Bregman metrics."""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "bregman_dual_metric_squared_hessian_v1"
CGUAGE_HESSIAN_EQUATION_ID = "cgauge_categorical_bregman_hessian_covariance_v1"
CLOSED_FORM_EQUATION_ID = "bregman_closed_form_dual_cancellation_v1"
NONNEGATIVITY_EQUATION_ID = "bregman_nonnegative_convexity_invariant_v1"
RIGHT_CENTROID_EQUATION_ID = "bregman_right_data_centroid_dual_mean_v1"
SIGMA_PROPAGATION_EQUATION_ID = "bregman_positive_unscented_propagation_v1"
APPLICATION_EQUATION_IDS = (
    CGUAGE_HESSIAN_EQUATION_ID,
    CLOSED_FORM_EQUATION_ID,
    NONNEGATIVITY_EQUATION_ID,
    RIGHT_CENTROID_EQUATION_ID,
    SIGMA_PROPAGATION_EQUATION_ID,
)
METRIC_ID = "argmax_native_vjp_fidelity_v1"
AXIS = "MEASURED_LOCAL_CPU_SYNTHETIC_MATH_FIXTURE_NOT_SCORE"
REAL_N600_SELECTION_STATUS = "NO_VERDICT_DATA_CUSTODY"
MEASUREMENT_UTC = "2026-07-14T14:16:00Z"
MEASUREMENT_ARTIFACT = ".omx/research/bregman_v9_all_surfaces_measurement_20260714.json"
BINDING_ARTIFACT = ".omx/research/bregman_v9_all_surfaces_binding_20260714.json"
INFORMATION_GEOMETRY_HELPER = "tac.information_geometry.bregman_v9_surfaces"
PREMISE_MEMO = (
    ".omx/research/"
    "codex_premise_falsification_bregman_dual_euclidean_20260714_codex.md"
)
APPLICATION_MEMO = ".omx/research/bregman_all_surfaces_504_derivation_20260715.md"
APPLICATION_DAG_FEED = ".omx/research/bregman_all_surfaces_504_DAG_FEED_20260715.md"
APPLICATION_UTC = "2026-07-15T02:00:00Z"
STATE_COUNT = 600
RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES = 600
MAX_PRIMAL_EXACT_DUAL_ERROR = 5.684341886080802e-14
MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR = 9.094947017729282e-13


def bregman_dual_metric_squared_hessian_law(
    hessian, delta_theta, delta_eta
) -> dict[str, float]:
    """Evaluate the four local quantities through the validated NumPy helper."""

    from tac.information_geometry.bregman_v9_surfaces import (
        local_hessian_dual_geometry_summary,
    )

    return local_hessian_dual_geometry_summary(hessian, delta_theta, delta_eta)


def build_bregman_dual_metric_squared_hessian_v1() -> CanonicalEquation:
    """Build the corrected equation and retained deterministic n600 anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEASUREMENT_ARTIFACT,
        reactivation_criteria=(
            "promotion is forbidden; real-n600 metric selection requires complete data "
            "custody and a separate authority-grade adoption decision"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU NumPy-fp64 synthetic SPD fixture",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="bregman_v9_dual_metric_correction_600_state_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "metric_id": METRIC_ID,
            "fixture": "synthetic_fixed_spd_local_chart_600_states",
            "state_count": STATE_COUNT,
            "dimension": 5,
            "axis": AXIS,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
        },
        predicted_output={
            "ordinary_hessian_identity": "dtheta.T@H@dtheta == deta.T@solve(H,deta)",
            "raw_dual_identity": "deta.T@deta == dtheta.T@H@H@dtheta",
            "raw_dual_equals_ordinary_hessian": False,
        },
        empirical_output={
            "raw_dual_vs_ordinary_hessian_mismatches": (
                RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES
            ),
            "state_count": STATE_COUNT,
            "max_primal_exact_dual_error": MAX_PRIMAL_EXACT_DUAL_ERROR,
            "max_raw_dual_squared_hessian_error": (
                MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR
            ),
            "dual_euclidean_no_solve_scope": "squared_hessian_H_squared_only",
            "binding_artifact": BINDING_ARTIFACT,
            "fisher_natural_cotangent_solve_elided": False,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
        },
        residual=MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR,
        source_artifact=MEASUREMENT_ARTIFACT,
        measurement_method=(
            "deterministic NumPy-fp64 evaluation of 600 five-dimensional SPD local charts; "
            "Fisher-natural cotangent evaluated with a typed linear solve"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Bregman dual metric squared-Hessian correction",
        one_line_summary=(
            "Fisher-natural cotangent length uses H^-1, while raw dual Euclidean length "
            "is the distinct H^2 primal quadratic form."
        ),
        latex_form=(
            r"\Delta\eta=H\Delta\theta,\quad "
            r"\Delta\theta^T H\Delta\theta="
            r"\Delta\eta^T H^{-1}\Delta\eta,\quad "
            r"\|\Delta\eta\|_2^2="
            r"\Delta\theta^T H^2\Delta\theta"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bregman_v9_surfaces_20260714:"
            "bregman_dual_metric_squared_hessian_law"
        ),
        domain_of_validity={
            "metric_id": METRIC_ID,
            "included": "finite non-empty local charts with symmetric positive-definite H",
            "fisher_natural_cotangent_geometry": "inverse_hessian_H_inverse",
            "fisher_natural_cotangent_solve": "typed_linear_solve",
            "fisher_natural_cotangent_solve_elided": False,
            "dual_euclidean_no_solve_scope": "squared_hessian_H_squared_only",
            "binding_artifact": BINDING_ARTIFACT,
            "axis": AXIS,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "excluded": (
                "non-SPD or non-finite metrics; raw dual Euclidean aliases for the "
                "Fisher-natural cotangent metric; score and promotion claims"
            ),
            "verdict_scope": (
                "prompt identity falsified for general SPD H; squared-Hessian form valid; "
                "canonical metric family not rejected"
            ),
        },
        units_in={
            "hessian": "dual_coordinate_units_per_primal_coordinate_unit",
            "delta_theta": "primal_coordinate_units",
            "delta_eta": "dual_coordinate_units",
        },
        units_out={
            "primal_hessian": "local_metric_squared_distance",
            "fisher_natural_cotangent": "local_metric_squared_distance",
            "raw_dual_euclidean": "dual_coordinate_squared_distance",
            "squared_hessian": "dual_coordinate_squared_distance",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "primal_vs_exact_dual": MAX_PRIMAL_EXACT_DUAL_ERROR,
            "raw_dual_vs_squared_hessian": MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.bregman_dual_metric_guard",
            "tac.witness_dsl.optimal_basis_20260714",
        ),
        canonical_producers=(INFORMATION_GEOMETRY_HELPER,),
        provenance=provenance,
    )


def populate_bregman_dual_metric_squared_hessian_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicitly register the equation; importing this module never mutates state."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_bregman_dual_metric_squared_hessian_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "Bregman dual-metric correction; synthetic math fixture only; "
            "promotion forbidden"
        ),
    )
    return equation


def cgauge_categorical_bregman_hessian_covariance_law(
    logits,
    *,
    point=None,
    reference=None,
    matrix=None,
    offset=None,
    scale=1.0,
    linear_term=None,
    constant=0.0,
):
    """Evaluate the categorical covariance Hessian and optional affine law."""

    import numpy as np

    from tac.information_geometry.bregman_v9_surfaces import (
        affine_legendre_logsumexp_summary,
        categorical_log_partition_hessian,
        categorical_softmax,
    )

    probability = categorical_softmax(logits)
    hessian = categorical_log_partition_hessian(logits)
    result = {
        "probability": probability,
        "hessian": hessian,
        "covariance": np.diag(probability) - np.outer(probability, probability),
        "ambient_null_residual": float(abs(hessian.sum(axis=1)).max()),
    }
    affine_inputs = (point, reference, matrix, offset, linear_term)
    if any(value is not None for value in affine_inputs):
        if not all(value is not None for value in affine_inputs):
            raise ValueError(
                "point, reference, matrix, offset, and linear_term must be supplied together"
            )
        result["affine_legendre"] = affine_legendre_logsumexp_summary(
            point,
            reference,
            matrix,
            offset,
            scale=scale,
            linear_term=linear_term,
            constant=constant,
        )
    return result


def bregman_closed_form_dual_cancellation_law(point_logits, reference_logits):
    """Evaluate exact solve-free divergence identities, never a raw-dual norm."""

    from tac.information_geometry.bregman_v9_surfaces import (
        logsumexp_bregman_closed_form_summary,
    )

    return logsumexp_bregman_closed_form_summary(point_logits, reference_logits)


def bregman_nonnegative_convexity_invariant_law(value, *, atol=1.0e-12):
    """Fail closed if a claimed convex-generator divergence is negative."""

    from tac.information_geometry.bregman_v9_surfaces import (
        require_nonnegative_bregman,
    )

    return require_nonnegative_bregman(value, atol=atol)


def bregman_right_data_centroid_dual_mean_law(sample_logits, weights):
    """Compute the right-data centroid and its quotient-gauge residual."""

    from tac.information_geometry.bregman_v9_surfaces import (
        categorical_right_data_centroid,
    )

    return categorical_right_data_centroid(sample_logits, weights)


def bregman_positive_unscented_propagation_law(
    mean, covariance, transform, *, kappa=1.0
):
    """Run the positive sigma rule and preserve its approximation boundary."""

    from tac.information_geometry.bregman_v9_surfaces import (
        categorical_bregman_sigma_propagation,
    )

    return categorical_bregman_sigma_propagation(
        mean, covariance, transform, kappa=kappa
    )


def _application_provenance():
    return build_provenance_for_research_sidecar(
        sidecar_path=APPLICATION_MEMO,
        reactivation_criteria=(
            "design law only; re-open trainer admission after a real consumer, typed "
            "DSL wire, and through-R acceptance receipt exist"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU NumPy-fp64 deterministic math fixture",
        captured_at_utc=APPLICATION_UTC,
    )


def _application_anchor(
    *,
    equation_id: str,
    claim: str,
    empirical_output: dict,
    residual: float,
) -> EmpiricalAnchor:
    provenance = _application_provenance()
    return EmpiricalAnchor(
        anchor_id=f"{equation_id}_deterministic_fixture_20260715",
        measurement_utc=APPLICATION_UTC,
        inputs={
            "metric_id": METRIC_ID,
            "fixture": "deterministic_numpy_fp64_three_class_and_sigma_rule",
            "axis": AXIS,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
        },
        predicted_output={"claim": claim, "residual_target": "<=1e-12"},
        empirical_output={
            **empirical_output,
            "score_claim": False,
            "promotion_eligible": False,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
        },
        residual=float(residual),
        source_artifact=APPLICATION_MEMO,
        measurement_method=(
            "deterministic NumPy-fp64 algebraic fixture; no scorer, archive, training, "
            "or contest evaluator"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _application_domain(**specific):
    return {
        "metric_id": METRIC_ID,
        "generator": "F(z)=logsumexp(z), K>=2 categorical logits",
        "ambient_metric": "positive_semidefinite_with_additive_logit_null",
        "quotient_metric": "positive_definite_after_additive_logit_gauge_fix",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
        "dsl_wire_status": "OWED_NO_EVIDENCED_TRAINER_CONSUMED_SWEPT_KNOB",
        **specific,
    }


def build_cgauge_categorical_bregman_hessian_covariance_v1() -> CanonicalEquation:
    """Build the exact categorical Hessian/covariance and affine pullback law."""

    from tac.information_geometry.bregman_v9_surfaces import (
        deterministic_bregman_application_fixture,
    )

    fixture = deterministic_bregman_application_fixture()
    residual = float(fixture["affine_gauge_covariance_error"])
    anchor = _application_anchor(
        equation_id=CGUAGE_HESSIAN_EQUATION_ID,
        claim="H_logsumexp=diag(p)-pp^T=Cov_p[e_Y] and affine pullback covariance",
        empirical_output={
            "affine_gauge_covariance_abs_error": residual,
            "categorical_covariance_identity": "DERIVED_EXACT",
        },
        residual=residual,
    )
    return CanonicalEquation(
        equation_id=CGUAGE_HESSIAN_EQUATION_ID,
        name="CGauge categorical Bregman Hessian and affine covariance",
        one_line_summary=(
            "The categorical CGauge output metric is exactly the logsumexp Bregman "
            "Hessian/covariance; full live scorer-pullback custody remains owed."
        ),
        latex_form=(
            r"F(z)=\log\sum_k e^{z_k},\ p=\nabla F,\ "
            r"\nabla^2F=\operatorname{diag}(p)-pp^T=\operatorname{Cov}_p[e_Y];\ "
            r"\bar F(\theta)=\lambda F(A\theta+b)+c^T\theta+d\Rightarrow "
            r"\nabla^2\bar F=\lambda A^T(\nabla^2F)A"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bregman_v9_surfaces_20260714:"
            "cgauge_categorical_bregman_hessian_covariance_law"
        ),
        domain_of_validity=_application_domain(
            cgauge_ground_metric_status=(
                "EXACT at categorical output; master-action A2 full frozen-scorer "
                "pullback and live affine-chart realization not established here"
            ),
            affine_legendre_model_status="IMPLEMENTATION_CUSTODY_GAP_ONLY",
            verdict_scope=(
                "categorical metric identity derived; absence of live V9 transform receipt "
                "is an implementation-custody gap, not a metric-family rejection"
            ),
        ),
        units_in={"logits": "categorical_logit_units"},
        units_out={"hessian": "categorical_fisher_curvature"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"affine_covariance": residual},
        last_calibration_utc=APPLICATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.canonical_equations.optimal_metric_unification_20260714",
            "tac.canonical_equations.cgauge_master_action_20260711",
        ),
        canonical_producers=(INFORMATION_GEOMETRY_HELPER,),
        provenance=_application_provenance(),
    )


def build_bregman_closed_form_dual_cancellation_v1() -> CanonicalEquation:
    """Build exact solve-free finite forms with a squared-Hessian exclusion."""

    from tac.information_geometry.bregman_v9_surfaces import (
        deterministic_bregman_application_fixture,
    )

    fixture = deterministic_bregman_application_fixture()
    residual = max(
        float(fixture["closed_form_dual_error"]),
        float(fixture["closed_form_cancellation_error"]),
    )
    anchor = _application_anchor(
        equation_id=CLOSED_FORM_EQUATION_ID,
        claim="Legendre-dual reversal and symmetrized gradient-pairing cancellation",
        empirical_output={
            "dual_identity_abs_error": fixture["closed_form_dual_error"],
            "cancellation_abs_error": fixture["closed_form_cancellation_error"],
            "fisher_natural_cotangent_solve_elided": False,
        },
        residual=residual,
    )
    return CanonicalEquation(
        equation_id=CLOSED_FORM_EQUATION_ID,
        name="Bregman exact dual reversal and gradient-pairing cancellation",
        one_line_summary=(
            "Finite Bregman/KL and its dual/cancellation forms are solve-free; raw "
            "dual Euclidean length remains the distinct squared-Hessian form."
        ),
        latex_form=(
            r"B_F(x\|y)=B_{F^*}(\nabla F(y)\|\nabla F(x));\quad "
            r"B_F(x\|y)+B_F(y\|x)=\langle\nabla F(x)-\nabla F(y),x-y\rangle"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bregman_v9_surfaces_20260714:"
            "bregman_closed_form_dual_cancellation_law"
        ),
        domain_of_validity=_application_domain(
            finite_logsumexp_identity="B_F(zS||zT)=KL(softmax(zT)||softmax(zS))",
            solve_free_scope="exact finite divergence and Legendre/cancellation identities",
            local_primal_fisher="dtheta^T H dtheta",
            fisher_natural_cotangent="deta^T solve(H,deta); solve required",
            raw_dual_no_solve="||deta||^2=dtheta^T H^2 dtheta; squared Hessian only",
            squared_hessian_owner=EQUATION_ID,
            excluded="finite Fisher-Rao geodesic-distance claim or raw-dual Fisher alias",
        ),
        units_in={"point_logits": "categorical_logit_units"},
        units_out={"divergence": "nats"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"closed_form_max_abs_error": residual},
        last_calibration_utc=APPLICATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(EQUATION_ID,),
        canonical_producers=(INFORMATION_GEOMETRY_HELPER,),
        provenance=_application_provenance(),
    )


def build_bregman_nonnegative_convexity_invariant_v1() -> CanonicalEquation:
    """Build the executable non-negativity/strictness guard."""

    anchor = _application_anchor(
        equation_id=NONNEGATIVITY_EQUATION_ID,
        claim="B_F>=0 iff the differentiable generator obeys first-order convexity",
        empirical_output={
            "negative_beyond_tolerance_refused": True,
            "strictness_scope": "logsumexp equality only on additive-logit quotient",
        },
        residual=0.0,
    )
    return CanonicalEquation(
        equation_id=NONNEGATIVITY_EQUATION_ID,
        name="Bregman convexity non-negativity fail-closed invariant",
        one_line_summary=(
            "A claimed convex-generator divergence must be non-negative, with "
            "logsumexp equality interpreted on the additive-logit quotient."
        ),
        latex_form=(
            r"F\text{ convex}\Longleftrightarrow "
            r"B_F(x\|y)=F(x)-F(y)-\langle\nabla F(y),x-y\rangle\ge0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bregman_v9_surfaces_20260714:"
            "bregman_nonnegative_convexity_invariant_law"
        ),
        domain_of_validity=_application_domain(
            included="finite in-domain differentiable generator evaluations",
            tolerance="only sub-1e-12 negative floating residue is clamped",
            strictness=(
                "strict convexity gives equality iff points agree; logsumexp agrees "
                "modulo an additive constant"
            ),
        ),
        units_in={"value": "divergence_units"},
        units_out={"validated_value": "nonnegative_divergence_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"guard_contract": 0.0},
        last_calibration_utc=APPLICATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(),
        canonical_producers=(INFORMATION_GEOMETRY_HELPER,),
        provenance=_application_provenance(),
    )


def build_bregman_right_data_centroid_dual_mean_v1() -> CanonicalEquation:
    """Build the right-data centroid with explicit orientation and gauge."""

    from tac.information_geometry.bregman_v9_surfaces import (
        deterministic_bregman_application_fixture,
    )

    fixture = deterministic_bregman_application_fixture()
    residual = float(fixture["centroid_first_order_residual"])
    anchor = _application_anchor(
        equation_id=RIGHT_CENTROID_EQUATION_ID,
        claim="argmin_c sum_i w_i B_F(c||theta_i)=gradF^-1(sum_i w_i gradF(theta_i))",
        empirical_output={
            "first_order_residual_linf": residual,
            "orientation": "samples_in_right_argument",
        },
        residual=residual,
    )
    return CanonicalEquation(
        equation_id=RIGHT_CENTROID_EQUATION_ID,
        name="Bregman right-data centroid as a dual-coordinate mean",
        one_line_summary=(
            "With samples in the right argument, the unique quotient-gauge centroid "
            "is inverse-gradient of the weighted dual mean."
        ),
        latex_form=(
            r"\arg\min_c\sum_iw_iB_F(c\|\theta_i)="
            r"(\nabla F)^{-1}\!\left(\sum_iw_i\nabla F(\theta_i)\right)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bregman_v9_surfaces_20260714:"
            "bregman_right_data_centroid_dual_mean_law"
        ),
        domain_of_validity=_application_domain(
            weights="strictly positive and normalized internally",
            orientation="samples occupy right argument B_F(c||theta_i)",
            opposite_orientation=(
                "samples-left B_F(theta_i||c) has the primal arithmetic mean; "
                "it is a different centroid"
            ),
            uniqueness="unique on additive-logit quotient; zero-mean gauge returned",
        ),
        units_in={"sample_logits": "N_by_K_categorical_logit_units"},
        units_out={"centroid_logits_zero_mean": "categorical_logit_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"first_order_linf": residual},
        last_calibration_utc=APPLICATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(),
        canonical_producers=(INFORMATION_GEOMETRY_HELPER,),
        provenance=_application_provenance(),
    )


def build_bregman_positive_unscented_propagation_v1() -> CanonicalEquation:
    """Build positive sigma moment matching with an honest nonlinear scope."""

    from tac.information_geometry.bregman_v9_surfaces import (
        deterministic_bregman_application_fixture,
    )

    fixture = deterministic_bregman_application_fixture()
    residual = max(
        float(fixture["sigma_input_mean_error"]),
        float(fixture["sigma_input_covariance_error"]),
        abs(float(fixture["ef_exact_condition_error"])),
    )
    anchor = _application_anchor(
        equation_id=SIGMA_PROPAGATION_EQUATION_ID,
        claim="positive 2D+1 sigma rule exactly matches input mean/covariance",
        empirical_output={
            "input_mean_abs_error": fixture["sigma_input_mean_error"],
            "input_covariance_abs_error": fixture["sigma_input_covariance_error"],
            "ef_expectation_match_error": fixture["ef_exact_condition_error"],
            "nonlinear_output_status": "APPROXIMATE",
        },
        residual=residual,
    )
    return CanonicalEquation(
        equation_id=SIGMA_PROPAGATION_EQUATION_ID,
        name="Positive unscented propagation under categorical Bregman geometry",
        one_line_summary=(
            "A positive 2D+1 rule matches input moments exactly and collapses mapped "
            "points by Bregman centroid; nonlinear output integration remains approximate."
        ),
        latex_form=(
            r"\chi_i^\pm=\mu\pm\sqrt{D+\kappa}L_{:i},\ "
            r"w_0=\kappa/(D+\kappa),\ w_i^\pm=1/[2(D+\kappa)],\ \kappa>0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bregman_v9_surfaces_20260714:"
            "bregman_positive_unscented_propagation_law"
        ),
        domain_of_validity=_application_domain(
            input_moment_status="EXACT for SPD covariance and kappa>0",
            transformed_centroid_status="EXACT for the selected transformed sigma support",
            nonlinear_output_status="APPROXIMATE for the full transformed distribution",
            exponential_family_exactness=(
                "KL quadrature exact only when sufficient-statistic expectation matches; "
                "error=(theta_p-theta_q)^T(eta_hat-eta_p)"
            ),
            kappa_lever_status="EQ_ONLY_NOT_A_REAL_TRAINER_CONSUMED_SWEEP",
        ),
        units_in={"mean": "input_coordinate_units", "covariance": "input_units_squared"},
        units_out={"bregman_dispersion": "nats"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"input_moment_max_abs_error": residual},
        last_calibration_utc=APPLICATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(),
        canonical_producers=(INFORMATION_GEOMETRY_HELPER,),
        provenance=_application_provenance(),
    )


def build_bregman_cgauge_application_equations_v1() -> tuple[CanonicalEquation, ...]:
    """Build all task-#504 application equations without mutating the registry."""

    return (
        build_cgauge_categorical_bregman_hessian_covariance_v1(),
        build_bregman_closed_form_dual_cancellation_v1(),
        build_bregman_nonnegative_convexity_invariant_v1(),
        build_bregman_right_data_centroid_dual_mean_v1(),
        build_bregman_positive_unscented_propagation_v1(),
    )


def populate_bregman_cgauge_application_equations_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> tuple[CanonicalEquation, ...]:
    """Append all task-#504 equations through the canonical locked registry API."""

    from tac.canonical_equations.registry import register_canonical_equation

    equations = build_bregman_cgauge_application_equations_v1()
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=(
                "task #504 Bregman V9-CGauge application; equations-only; DSL wires owed; "
                "no score/promotion/training claim"
            ),
        )
    return equations


__all__ = [
    "APPLICATION_DAG_FEED",
    "APPLICATION_EQUATION_IDS",
    "APPLICATION_MEMO",
    "APPLICATION_UTC",
    "AXIS",
    "BINDING_ARTIFACT",
    "CGUAGE_HESSIAN_EQUATION_ID",
    "CLOSED_FORM_EQUATION_ID",
    "EQUATION_ID",
    "INFORMATION_GEOMETRY_HELPER",
    "MAX_PRIMAL_EXACT_DUAL_ERROR",
    "MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR",
    "METRIC_ID",
    "NONNEGATIVITY_EQUATION_ID",
    "RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES",
    "REAL_N600_SELECTION_STATUS",
    "RIGHT_CENTROID_EQUATION_ID",
    "SIGMA_PROPAGATION_EQUATION_ID",
    "STATE_COUNT",
    "bregman_closed_form_dual_cancellation_law",
    "bregman_dual_metric_squared_hessian_law",
    "bregman_nonnegative_convexity_invariant_law",
    "bregman_positive_unscented_propagation_law",
    "bregman_right_data_centroid_dual_mean_law",
    "build_bregman_cgauge_application_equations_v1",
    "build_bregman_closed_form_dual_cancellation_v1",
    "build_bregman_dual_metric_squared_hessian_v1",
    "build_bregman_nonnegative_convexity_invariant_v1",
    "build_bregman_positive_unscented_propagation_v1",
    "build_bregman_right_data_centroid_dual_mean_v1",
    "build_cgauge_categorical_bregman_hessian_covariance_v1",
    "cgauge_categorical_bregman_hessian_covariance_law",
    "populate_bregman_cgauge_application_equations_v1",
    "populate_bregman_dual_metric_squared_hessian_v1",
]
