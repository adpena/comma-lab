# SPDX-License-Identifier: MIT
"""Canonical laws from the 2026-07-13 manifold/geometry slot audit.

The module deliberately does not modify the shared package import table.  Its
``populate_*`` entrypoint registers all three laws through the canonical
fcntl-locked writer; this keeps the anti-collision lane additive while still
making the equations queryable from the canonical registry.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

UTC = "2026-07-13T18:15:00Z"
MEMO = ".omx/research/manifold_geometry_slots_dig_20260713.md"
RECEIPT = ".omx/research/manifold_geometry_slots_probe_receipt_20260713.json"
STAGE_S1_S2 = ".omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json"
AXIS = "[macOS-CPU numpy advisory; n600 cached artifacts; non-promotable]"


def equal_flip_metric_density(flip_density: Iterable[float]) -> tuple[float, ...]:
    """Return metric arc-length density that equalizes first-order flip mass.

    For row risk ``w(v)`` and ``N`` uniform metric bins, local bin width is
    ``dv ~= 1/(N rho(v))``.  Equal first-order risk ``w(v) dv`` therefore
    requires ``rho(v)=sqrt(g_vv) proportional to w(v)``.
    """

    values = tuple(float(value) for value in flip_density)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("flip_density must be a non-empty finite non-negative sequence")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("flip_density must have positive mass")
    return tuple(value / total for value in values)


def fisher_pairwise_wall_distance(logit_gap: float) -> float:
    """Exact Fisher--Rao distance to the top-two decision wall.

    The two classes are renormalized to sum to one.  A top1-top2 logit gap does
    not determine the other three class probabilities, so this helper refuses
    to call the result a full five-class simplex distance.
    """

    gap = float(logit_gap)
    if not math.isfinite(gap):
        raise ValueError("logit_gap must be finite")
    p1 = 1.0 / (1.0 + math.exp(-min(max(gap, -700.0), 700.0)))
    p2 = 1.0 - p1
    argument = abs(math.sqrt(p1) - math.sqrt(p2)) / math.sqrt(2.0)
    return 2.0 * math.asin(min(max(argument, 0.0), 1.0))


def advective_acoustic_metric(flow_xy: Sequence[float], residual_speed: float) -> tuple[tuple[float, ...], ...]:
    """Lorentzian envelope ``||dx-u dt||^2-c^2 dt^2`` in ``(t,x,y)``.

    This metric is admissible only when a finite isotropic residual propagation
    speed ``c`` is part of the model.  Pure deterministic advection retains the
    weaker Newton--Cartan/Galilean structure and does not mint a light cone.
    """

    if len(flow_xy) != 2:
        raise ValueError("flow_xy must contain (u_x,u_y)")
    ux, uy = (float(value) for value in flow_xy)
    c = float(residual_speed)
    if not all(math.isfinite(value) for value in (ux, uy, c)) or c <= 0.0:
        raise ValueError("flow and residual_speed must be finite, with residual_speed > 0")
    return (
        (ux * ux + uy * uy - c * c, -ux, -uy),
        (-ux, 1.0, 0.0),
        (-uy, 0.0, 1.0),
    )


def worldsheet_rate_saving(
    independent_frame_entropy: Iterable[float],
    initial_curve_entropy: float,
    event_entropy: Iterable[float],
    phase_entropy: Iterable[float],
    event_residual_entropy: Iterable[float],
) -> float:
    """Derived independent-frame minus marked-world-sheet codelength.

    A positive result is a rate opportunity, not a guaranteed archive saving;
    receiver/model overhead and exact decoded-state parity still gate admission.
    """

    independent = tuple(float(value) for value in independent_frame_entropy)
    event = tuple(float(value) for value in event_entropy)
    phase = tuple(float(value) for value in phase_entropy)
    residual = tuple(float(value) for value in event_residual_entropy)
    initial = float(initial_curve_entropy)
    all_values = (*independent, initial, *event, *phase, *residual)
    if any(not math.isfinite(value) or value < 0.0 for value in all_values):
        raise ValueError("all entropy terms must be finite and non-negative")
    if not (len(event) == len(phase) == len(residual)):
        raise ValueError("marked temporal entropy streams must have equal length")
    return sum(independent) - (initial + sum(event) + sum(phase) + sum(residual))


def build_flip_density_chart_metric_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="n600_row_flip_density_chart_comparison_20260713",
        measurement_utc=UTC,
        inputs={
            "source": ".omx/research/dseg_reducibility_gt_margin_n600_20260623.json",
            "n_pairs": 600,
            "rows": "v>174",
            "scope": "all-class flip density in the ground-support row domain",
        },
        predicted_output={
            "law": "sqrt(g_vv)=rho(v) proportional to w_flip(v)",
            "candidates": ["uniform", "1/(v-vh)", "1/(v-vh)^2"],
        },
        empirical_output={
            "flip_mass_below_horizon": 0.9292588586095267,
            "peak_row": 193,
            "js_uniform": 0.24818817978749463,
            "js_log_depth": 0.16044044809089875,
            "js_inverse_depth": 0.46026082706106153,
            "js_shifted_inverse_depth": 0.06994264689610602,
            "shifted_inverse_depth_offset_rows": 32.5257801441824,
            "verdict_scope": "n600 all-class row-density proxy; ground-class-only density remains unmeasured",
        },
        residual=0.0,
        source_artifact=STAGE_S1_S2,
        measurement_method="n600_cached_row_flip_density_companding_comparison",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            STAGE_S1_S2,
            "remeasure a ground-class-pair-only row ledger and require a byte-closed chart A/B before promotion",
            AXIS,
            "macos_arm64_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id="flip_density_chart_metric_v1",
        name="Measured flip-density companding metric for witness row charts",
        one_line_summary="Uniform chart arc length should follow measured row flip density; log-depth beats uniform and raw inverse-depth on the n600 proxy.",
        latex_form=(
            r"d\ell=\rho(v)\,dv,\quad \rho^*(v)=\sqrt{g_{vv}(v)}="
            r"\frac{w_{flip}(v)}{\int w_{flip}(s)ds},\quad w(v)\Delta v=\mathrm{const}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.manifold_geometry_slots_20260713:equal_flip_metric_density"
        ),
        domain_of_validity={
            "included": "first-order equal-risk row companding over the measured v>174 support",
            "excluded": "class-pair routing, interpolation-order optimum, and score promotion",
            "verdict_scope": "FORMULATION x SNAPSHOT: n600 all-class row-risk proxy",
        },
        units_in={"flip_density": "flip_probability_per_image_row"},
        units_out={"metric_density": "normalized_inverse_rows"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"equal_mass_identity": 0.0},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.boundary_math.ground_frame_chart", "tac.witness_dsl.curriculum_candidate_pool"),
        canonical_producers=("tools.probe_manifold_geometry_slots", STAGE_S1_S2),
        provenance=build_provenance_for_research_sidecar(
            MEMO,
            "a default-off compander must beat the #194 chart at n600 and survive receiver-close before promotion",
            AXIS,
            "macos_arm64_cpu",
        ),
    )


def build_fisher_pairwise_decision_wall_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="n600_flip_margin_fisher_flat_shadow_20260713",
        measurement_utc=UTC,
        inputs={"n_pairs": 600, "n_flip_margins": 250_519, "source": STAGE_S1_S2},
        predicted_output={"local_flat_shadow": "d_FR,pair(delta)=delta/2+O(delta^3)"},
        empirical_output={
            "flip_median_gap": 0.12164878845214844,
            "flip_median_relative_error": -0.000616031472169265,
            "flip_p90_gap": 0.4399970054626466,
            "flip_p90_relative_error": -0.007970304445913068,
            "strictly_monotone": True,
            "verdict_scope": "top-two renormalized pair; full five-class distance unavailable from gap-only cache",
        },
        residual=0.007970304445913068,
        source_artifact=STAGE_S1_S2,
        measurement_method="n600_flip_margin_pairwise_fisher_flat_shadow",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            STAGE_S1_S2,
            "retain ETF as incumbent unless a matched n600 full-logit geodesic-head A/B moves d_seg",
            AXIS,
            "macos_arm64_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id="fisher_pairwise_decision_wall_v1",
        name="Fisher--Rao top-two simplex decision-wall distance",
        one_line_summary="The exact pairwise Fisher wall distance is monotone in the logit gap and differs from gap/2 by under 0.8% through the n600 flip p90.",
        latex_form=(
            r"d_{FR,2}(\delta)=2\arcsin\!\left(\frac{|\sqrt{\sigma(\delta)}-"
            r"\sqrt{\sigma(-\delta)}|}{\sqrt2}\right)=\frac{|\delta|}{2}+O(|\delta|^3)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.manifold_geometry_slots_20260713:fisher_pairwise_wall_distance"
        ),
        domain_of_validity={
            "included": "top-two probabilities renormalized to a binary simplex",
            "excluded": "full K=5 Fisher distance from gap alone; training-gradient or score claims",
            "verdict_scope": "GEOMETRY: local decision-wall metric on the measured flip-margin range",
        },
        units_in={"logit_gap": "dimensionless_logits"},
        units_out={"distance": "Fisher_Rao_radians_radius_2"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"flat_shadow_relative_error_at_flip_p90": 0.007970304445913068},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.curriculum_candidate_pool",),
        canonical_producers=("tools.probe_manifold_geometry_slots", STAGE_S1_S2),
        provenance=build_provenance_for_research_sidecar(
            MEMO,
            "full K=5 cached logits plus a matched ETF-versus-geodesic loss A/B are required to reopen",
            AXIS,
            "macos_arm64_cpu",
        ),
    )


def build_advective_worldsheet_rate_v1() -> CanonicalEquation:
    return CanonicalEquation(
        equation_id="advective_worldsheet_rate_v1",
        name="Galilean advection, optional acoustic cone, and marked world-sheet rate",
        one_line_summary="Pure xi transport is Galilean; a finite residual speed induces an acoustic Lorentzian cone, while world-sheet savings equal conditional-entropy reduction.",
        latex_form=(
            r"ds^2=\|d\mathbf x-\mathbf u dt\|^2-c^2dt^2;\quad "
            r"\Delta R=\sum_tH(\Gamma_t|C)-[H(\Gamma_0|C)+\sum_t(H(E_t|X_t,C)+"
            r"H(\Phi_t|E_t,X_t,C)+H(\Delta_t^E|\Phi_t,E_t,X_t,C))]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.manifold_geometry_slots_20260713:worldsheet_rate_saving"
        ),
        domain_of_validity={
            "primary_structure": "Newton-Cartan/Galilean absolute time plus xi advection field",
            "lorentzian_condition": "finite isotropic residual propagation speed c>0 is explicitly modeled",
            "de_sitter_excluded": "no measured constant positive curvature or SO(4,1) symmetry",
            "rate_gate": "identical decoded world-sheets and receiver/model overhead included",
            "verdict_scope": "DERIVED chain rule; byte advantage remains unmeasured for the separatrix codec",
        },
        units_in={"entropy_terms": "bits", "flow": "pixels_per_pair", "residual_speed": "pixels_per_pair"},
        units_out={"rate_saving": "bits", "metric": "coordinate_quadratic_form"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=("tac.canonical_equations.rate_law_ladder_20260713", "tac.witness_dsl.curriculum_candidate_pool"),
        canonical_producers=(MEMO,),
        provenance=build_provenance_for_research_sidecar(
            MEMO,
            "run an identical-decode per-frame versus marked-world-sheet n600 byte A/B before any rate claim",
            "[DERIVED theory; no empirical byte claim]",
            "not_applicable",
        ),
    )


def build_all_manifold_geometry_slot_equations() -> tuple[CanonicalEquation, ...]:
    return (
        build_flip_density_chart_metric_v1(),
        build_fisher_pairwise_decision_wall_v1(),
        build_advective_worldsheet_rate_v1(),
    )


def populate_manifold_geometry_slot_equations(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> tuple[CanonicalEquation, ...]:
    """Register the three laws through the canonical fcntl-locked API."""

    from tac.canonical_equations.registry import register_canonical_equation

    equations = build_all_manifold_geometry_slot_equations()
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes="FEED-manifold-slots-20260713; score_claim=false; pointer_moved=false",
        )
    return equations


__all__ = [
    "advective_acoustic_metric",
    "build_advective_worldsheet_rate_v1",
    "build_all_manifold_geometry_slot_equations",
    "build_fisher_pairwise_decision_wall_v1",
    "build_flip_density_chart_metric_v1",
    "equal_flip_metric_density",
    "fisher_pairwise_wall_distance",
    "populate_manifold_geometry_slot_equations",
    "worldsheet_rate_saving",
]
