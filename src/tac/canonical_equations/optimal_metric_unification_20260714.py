# SPDX-License-Identifier: MIT
"""The OPTIMAL METRIC unification: one ``g = ∇²F`` serving fidelity, loss, curriculum.

Operator task #500 (2026-07-14): unify the ALREADY-LANDED information-geometry
pieces into ONE reachable-decision-geometry metric that is simultaneously the
fidelity predicate, the training-loss geometry, and a curriculum-varying metric.
This module registers two canonical equations:

* ``optimal_metric_unification_v1`` — the single metric ``g = ∇²F(θ)`` (log-partition
  Hessian = categorical Fisher = the Bregman Hessian of ``F = logsumexp``) and the
  DERIVED reductions to each role. Each reduction is computed (not asserted) in
  ``tac.information_geometry.optimal_metric``; the tests prove the fidelity reduction
  is bit-equal to the RIPO curvature.
* ``categorical_fisher_trust_region_winner_rival_v1`` — the OWED RIPO directional
  trust-region law ``|t| ≤ √(8·δ_KL / C_wr)``, ``C_wr = p_w + p_r − (p_w − p_r)²``
  (a directional quadratic form of the SAME ``g``), which replaces the FALSE binary
  intake transfer ``√(δ/p_w)`` per the MEASURED Spearman −0.96 falsification.

Both are DERIVED laws grounded in the Bregman framework (Nielsen). NO-FAKE: the
metric is used in its PRIMAL / tangent (logit-displacement) quadratic form ``Δθᵀ g Δθ``
throughout — that IS Fisher-natural with no ``H⁻¹`` solve. The DUAL raw-mean no-solve
length is the SQUARED Hessian ``Δθᵀ g² Δθ`` and is NOT the Fisher-natural cotangent
length (which needs an ``H⁻¹`` solve) — the landed guard
``bregman_dual_metric_squared_hessian_v1`` owns that distinction and this module
honors it (never conflates squared-Hessian with Fisher-natural).
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

METRIC_EVALUATOR = "tac.information_geometry.optimal_metric"
MEASUREMENT_UTC = "2026-07-14T16:40:00Z"
AXIS = "MEASURED_LOCAL_CPU_NUMPY_FP32_ADVISORY_NOT_SCORE"

# --- fidelity anchor: the RIPO MEASURED directional-vs-binary falsification ---
RIPO_RECEIPT = ".omx/research/ripo_categorical_fisher_binary_vs_directional_measured_20260714.json"
RIPO_MEMO = ".omx/research/ripo_categorical_fisher_binary_vs_directional_MEASURED_20260714.md"
RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL = -0.9601  # MEASURED (real SegNet K=5, 18.87M px)
RIPO_RATIO_DIR_OVER_BIN_MEDIAN = 16.34  # MEASURED median r_dir/r_bin
RIPO_ARGMAX_MATCH_FRACTION = 1.000000  # reproduced logits == cached lstars (authority forward)

# --- calibration anchor: the margin<->Fisher caustic (already registered) ---
MARGIN_FISHER_ANCHOR = "curvature_neg_margin_pearson_0978_spearman_0908_caustic_20260704"
MARGIN_FISHER_PEARSON = 0.978  # MEASURED-ANCHOR (deepmath_amortizing_argmax_laws)
MARGIN_FISHER_SPEARMAN = 0.908
UNIFICATION_MEMO = ".omx/research/optimal_metric_unification_derivation_20260714.md"


def optimal_metric_unification_law(logits, tau=1.0, direction=None):
    """Evaluate the three role readings of ``g = ∇²F`` through the validated helper."""

    from tac.information_geometry.optimal_metric import (
        optimal_metric_unification_law as _law,
    )

    return _law(logits, tau=tau, direction=direction)


def winner_rival_trust_radius(probabilities, *, delta, delta_convention="delta_kl"):
    """Evaluate the RIPO directional trust radius ``|t| = √(8·δ_KL / C_wr)`` through
    the existing validated implementation (no duplication)."""

    from tac.optimization.ripo_fisher_trust_region import winner_rival_radius

    return winner_rival_radius(
        probabilities, delta=delta, delta_convention=delta_convention
    )


def build_optimal_metric_unification_v1() -> CanonicalEquation:
    """The single metric ``g = ∇²F`` + its DERIVED fidelity/loss/curriculum reductions."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=UNIFICATION_MEMO,
        reactivation_criteria=(
            "recalibrate when a new measured margin<->Fisher band anchor OR a new "
            "directional trust-region receipt lands; promotion forbidden (design law)"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU NumPy-fp32; derived law + cited measured anchors",
        captured_at_utc=MEASUREMENT_UTC,
    )
    # Anchor 1 (fidelity reduction, bit-verified): C_wr = (e_w-e_r)^T g (e_w-e_r).
    fidelity_anchor = EmpiricalAnchor(
        anchor_id="optimal_metric_fidelity_reduction_cwr_bit_equal_ripo_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "reduction": "fidelity",
            "identity": "C_wr = (e_w-e_r)^T (diag(p)-p p^T) (e_w-e_r) = p_w+p_r-(p_w-p_r)^2",
            "verified_against": "tac.optimization.ripo_fisher_trust_region.winner_rival_curvature",
            "axis": AXIS,
        },
        predicted_output={
            "metric_directional_quadratic_equals_ripo_curvature": True,
            "requires_h_inverse_solve": False,
        },
        empirical_output={
            "bit_equal": True,
            "note": "primal/tangent quadratic form; Fisher-natural in logit coords; no H^-1 solve",
            "ripo_spearman_binary_vs_directional": RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL,
            "score_claim": False,
            "promotion_eligible": False,
        },
        residual=0.0,
        source_artifact=RIPO_RECEIPT,
        measurement_method=(
            "numpy-fp64 bit-equality of the directional quadratic (e_w-e_r)^T g (e_w-e_r) "
            "against the independently-implemented RIPO winner_rival_curvature"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    # Anchor 2 (training-loss reduction, measured calibration): margin<->Fisher 0.978.
    loss_anchor = EmpiricalAnchor(
        anchor_id="optimal_metric_trainingloss_reduction_margin_is_fisher_0978_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "reduction": "training_loss",
            "identity": "tr g|2-class = 2 p(1-p) = 1/2 sech^2(m/2), p=sigma(m)",
            "calibration_anchor": MARGIN_FISHER_ANCHOR,
            "axis": AXIS,
        },
        predicted_output={
            "margin_field_is_fisher_surrogate": True,
            "exact_global_identity": False,
            "scope": "two-class annulus scalar TRACE surrogate; measured band calibration",
        },
        empirical_output={
            "pearson_band": MARGIN_FISHER_PEARSON,
            "spearman_global": MARGIN_FISHER_SPEARMAN,
            "honest_gap": (
                "the witness descends the SCALAR margin surrogate for tr(g), not the full "
                "K=5 directional metric g; the reduction is a measured 0.978 band calibration, "
                "not an exact identity"
            ),
            "score_claim": False,
            "promotion_eligible": False,
        },
        residual=1.0 - MARGIN_FISHER_PEARSON,
        source_artifact=(
            "src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py"
        ),
        measurement_method=(
            "cites the registered caustic anchor "
            f"{MARGIN_FISHER_ANCHOR} (curvature<->(-margin) Pearson 0.978)"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id="optimal_metric_unification_v1",
        name="Optimal metric unification (g = del^2 F: fidelity, loss, curriculum)",
        one_line_summary=(
            "One metric g=del^2 F (categorical Fisher) reduces to fidelity (directional C_wr), "
            "training-loss (margin=Fisher 0.978 surrogate), and curriculum (tau-tempered g)."
        ),
        latex_form=(
            r"g(\theta)=\nabla^2 F=\mathrm{diag}(p)-pp^T,\ p=\mathrm{softmax}(\theta);\ "
            r"\text{fidelity: }C_{wr}=(e_w-e_r)^T g (e_w-e_r)=p_w+p_r-(p_w-p_r)^2;\ "
            r"\text{loss: }\mathrm{tr}\,g|_{2}=\tfrac12\mathrm{sech}^2(m/2);\ "
            r"\text{curriculum: }g(\tau)=\tau^{-2}(\mathrm{diag}(p_\tau)-p_\tau p_\tau^T)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.optimal_metric_unification_20260714:"
            "optimal_metric_unification_law"
        ),
        domain_of_validity={
            "generator": "F(theta)=logsumexp(theta); Bregman/log-partition of K>=2 softmax",
            "metric_form_used": "PRIMAL tangent quadratic dtheta^T g dtheta (Fisher-natural, no H^-1)",
            "dual_no_solve_is_squared_hessian_not_fisher_natural": True,
            "squared_hessian_owner": "bregman_dual_metric_squared_hessian_v1",
            "fidelity_reduction": "EXACT (directional quadratic form; bit-verified vs RIPO C_wr)",
            "training_loss_reduction": (
                "PARTIAL SURROGATE (scalar two-class trace; measured 0.978 band calibration, "
                "NOT an exact global identity)"
            ),
            "curriculum_reduction": (
                "DERIVED (g(tau)=tau^-2 (diag(p_tau)-p_tau p_tau^T); prefactor via chain rule, "
                "operating point via softmax(theta/tau); concentrates on separatrix as tau down)"
            ),
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": (
                "unification of measured/derived pieces; fidelity reduction exact, training-loss "
                "reduction is an honest measured surrogate, curriculum reduction derived"
            ),
        },
        units_in={
            "logits": "categorical_logit_units",
            "tau": "temperature_dimensionless",
            "direction": "logit_displacement_units_or_none",
        },
        units_out={
            "fidelity_directional_curvature": "kl_curvature_per_squared_logit_displacement",
            "training_loss_margin_surrogate_trace": "categorical_fisher_trace_dimensionless",
            "curriculum_tau_reading": "tempered_curvature_and_operating_point",
        },
        empirical_anchors=(fidelity_anchor, loss_anchor),
        predicted_vs_empirical_residual={
            "fidelity_bit_equality": 0.0,
            "training_loss_margin_fisher_band": 1.0 - MARGIN_FISHER_PEARSON,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            # surrogate-admission (fidelity): the RIPO directional trust region
            "tac.optimization.ripo_fisher_trust_region",
            # training-loss (margin field surrogate)
            "tac.canonical_equations.horizon_weighted_margin_20260709",
            # curriculum (tau-varying metric) + covariant sister
            "tac.canonical_equations.cgauge_master_action_20260711",
        ),
        canonical_producers=(
            METRIC_EVALUATOR,
            "tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704",
            "tac.canonical_equations.bregman_v9_surfaces_20260714",
        ),
        provenance=provenance,
    )


def build_categorical_fisher_trust_region_winner_rival_v1() -> CanonicalEquation:
    """The OWED RIPO directional trust-region law (fidelity specialization of g)."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=RIPO_RECEIPT,
        reactivation_criteria=(
            "reopen surrogate re-admission only with RE-CAPTURED advanced-locus receipts "
            "(centered logits, probabilities, directional Jacobians); n=0 receipt is not a negative"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU NumPy-fp32; real SegNet K=5 n96 GT cache",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ripo_categorical_fisher_binary_vs_directional_falsified_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "false_binary_transfer": "||dlogit||/sqrt(delta) = 1/sqrt(p_w)",
            "correct_directional_law": "|t|/sqrt(delta_kl) = 2/sqrt(C_wr)",
            "C_wr": "p_w + p_r - (p_w - p_r)^2",
            "real_pixels": 18874368,
            "argmax_match_fraction": RIPO_ARGMAX_MATCH_FRACTION,
            "axis": AXIS,
        },
        predicted_output={
            "binary_transfer_valid_for_K5_softmax": False,
            "directional_law_is_correct_fisher_ball": True,
        },
        empirical_output={
            "spearman_binary_vs_directional": RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL,
            "ratio_dir_over_bin_median": RIPO_RATIO_DIR_OVER_BIN_MEDIAN,
            "binary_over_admit_fraction": 0.0,
            "verdict_scope": "FALSIFICATION-CONFIRMED at FORMULATION level (not Fisher/KL family)",
            "score_claim": False,
            "promotion_eligible": False,
        },
        residual=abs(RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL),
        source_artifact=RIPO_RECEIPT,
        measurement_method=(
            "real SegNet K=5 logits (n96 GT cache, 18.87M pixels; argmax==cached lstars 1.0); "
            "Spearman rank corr of r_bin=1/sqrt(p_w) vs r_dir=2/sqrt(C_wr)"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id="categorical_fisher_trust_region_winner_rival_v1",
        name="Categorical-Fisher directional trust region (winner-rival curvature)",
        one_line_summary=(
            "Correct K=5 softmax trust region |t|<=sqrt(8 delta_KL/C_wr), C_wr=p_w+p_r-(p_w-p_r)^2; "
            "replaces the FALSE binary sqrt(delta/p_w) (Spearman -0.96)."
        ),
        latex_form=(
            r"|t|\le\sqrt{8\,\delta_{KL}/C_{wr}},\quad "
            r"C_{wr}=p_w+p_r-(p_w-p_r)^2=(e_w-e_r)^T\big(\mathrm{diag}(p)-pp^T\big)(e_w-e_r)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.optimal_metric_unification_20260714:"
            "winner_rival_trust_radius"
        ),
        domain_of_validity={
            "applies_to": "K>=2 categorical softmax (K=5 SegNet) winner<->runner-up direction",
            "is_directional_quadratic_of": "the categorical Fisher metric g=diag(p)-p p^T",
            "false_transfer_rejected": "||dlogit||<=sqrt(delta/p_w) (binary/scalar-logit cargo-cult)",
            "operating_point": "real SegNet p_w median 0.994 (near-tie annulus tail)",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "FORMULATION-level falsification of the binary transfer; not the KL/Fisher family",
        },
        units_in={
            "probabilities": "categorical_probability_simplex",
            "delta": "kl_or_quadratic_budget_nats",
            "delta_convention": "delta_kl_or_delta_quad_token",
        },
        units_out={"winner_rival_trust_radius": "logit_difference_displacement_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "binary_vs_directional_spearman": abs(RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL),
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.canonical_equations.optimal_metric_unification_20260714",
        ),
        canonical_producers=(
            "tac.optimization.ripo_fisher_trust_region",
            METRIC_EVALUATOR,
        ),
        provenance=provenance,
    )


def populate_optimal_metric_unification_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> list[CanonicalEquation]:
    """Register BOTH equations idempotently; importing this module never mutates state."""

    from tac.canonical_equations.registry import register_canonical_equation

    equations = [
        build_categorical_fisher_trust_region_winner_rival_v1(),
        build_optimal_metric_unification_v1(),
    ]
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=(
                "optimal metric unification (task #500); derived law + cited measured anchors; "
                "promotion forbidden"
            ),
        )
    return equations


__all__ = [
    "AXIS",
    "MARGIN_FISHER_ANCHOR",
    "MARGIN_FISHER_PEARSON",
    "METRIC_EVALUATOR",
    "RIPO_RATIO_DIR_OVER_BIN_MEDIAN",
    "RIPO_RECEIPT",
    "RIPO_SPEARMAN_BINARY_VS_DIRECTIONAL",
    "build_categorical_fisher_trust_region_winner_rival_v1",
    "build_optimal_metric_unification_v1",
    "optimal_metric_unification_law",
    "populate_optimal_metric_unification_v1",
    "winner_rival_trust_radius",
]
