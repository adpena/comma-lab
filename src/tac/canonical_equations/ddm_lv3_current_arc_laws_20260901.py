# SPDX-License-Identifier: MIT
"""Current-arc DDM laws registered by the LV3 recursive-leverage wave.

The laws here are deliberately narrow, executable, and source-anchored.  They
do not turn advisory measurements into authority scores, transfer constants to
new vehicles, or reopen closed instances.  Decoder-causal conditioning is not
duplicated here: it is already registered as
``decoder_causal_condition_transport_v1`` by CCS1.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import (
    eval_byte_distortion_cross_intersection_count,
    eval_context_model_reorder_savings,
    eval_decoder_derivable_ideal_savings_ceiling,
    eval_field_change_bhw_decomposition,
    eval_roundtrip_token_to_argmax_affine,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO = Path(__file__).resolve().parents[3]

SHARP_ID = "same_basin_sharp_optimum_v1"
CROSS_ID = "byte_distortion_cross_intersection_count_v1"
AFFINE_ID = "roundtrip_token_to_argmax_affine_v1"
BHW_ID = "field_change_bhw_decomposition_v1"
REORDER_ID = "context_model_reorder_savings_v1"
GENERATOR_ID = "generator_form_fit_error_entanglement_v1"
CEILING_ID = "decoder_derivable_ideal_savings_ceiling_v1"

HC1 = ".omx/research/ddm_hc1_hpac_calibration_reliability_20260824.md"
NI1R = ".omx/research/ddm_ni1r_nr1_k32_distortion_measured_20260830.md"
X012 = ".omx/research/ddm_x012_crossing_ledger_20260901.md"
GF1 = ".omx/research/ddm_gf1_generator_form_capacity_verdict_20260830.md"
LBX1 = ".omx/research/ddm_lbx1_lb1_exchange_curve_20260831.md"
FCD1 = ".omx/research/ddm_fcd1_field_for_coder_diagonal_20260829.md"
DDS1 = ".omx/research/ddm_dds1_ceiling_readjudication_20260901.md"


def _prov(source: str, reactivation: str, *, axis: str = "[source-inspected advisory]"):
    return build_provenance_for_research_sidecar(
        REPO / source,
        reactivation_criteria=reactivation,
        measurement_axis=axis,
        hardware_substrate="source_inspection_and_retained_exact_coder_receipts",
        captured_at_utc="2026-09-01T22:40:00Z",
    )


def _anchor(
    *,
    anchor_id: str,
    source: str,
    inputs: dict[str, Any],
    predicted: dict[str, Any],
    empirical: dict[str, Any],
    residual: float,
    method: str,
    provenance,
    inspected: bool = False,
) -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc="2026-09-01T22:40:00Z",
        inputs=inputs,
        predicted_output=predicted,
        empirical_output=empirical,
        residual=float(residual),
        source_artifact=source,
        measurement_method=method,
        provenance=provenance,
        empirical_verification_status=(
            VERIFIED_VIA_SOURCE_INSPECTION if inspected else VERIFIED_VIA_EMPIRICAL_ANCHOR
        ),
    )


def build_same_basin_sharp_optimum_v1() -> CanonicalEquation:
    provenance = _prov(
        HC1,
        "recalibrate only when a content-distinct same-basin direction is measured or the basin definition changes",
        axis="[macOS-CPU advisory plus source inspection]",
    )
    # HC1 is the sixth direction-family confirmation.  The sources do not
    # publish a commensurate seven-value delta vector, so the anchor preserves
    # only the measured signs; the evaluator remains available for future rows
    # that do publish their exact deltas.
    anchor = _anchor(
        anchor_id="hc1_sixth_same_basin_direction_family_nonimproving_20260824",
        source=HC1,
        inputs={"direction_family_count": 6, "published_exact_delta_vector": False},
        predicted={"minimum_delta_s_nonnegative": True},
        empirical={"all_direction_families_nonimproving": True, "exact_minimum_delta_s": "NOT_PUBLISHED_AS_ONE_VECTOR"},
        residual=0.0,
        method="bounded source join across five predecessor direction families plus HC1 probability calibration",
        provenance=provenance,
        inspected=True,
    )
    return CanonicalEquation(
        equation_id=SHARP_ID,
        name="Same-basin sharp optimum admission",
        one_line_summary="A measured same-basin family is locally sharp when every admitted direction has non-negative objective displacement; this does not close a new basin.",
        latex_form=r"\Delta S_{\min}(\mathcal B)=\min_{d\in\mathcal B}\Delta S(d)\ge 0",
        python_callable_module_path="tac.canonical_equations.evaluators:eval_same_basin_sharp_optimum",
        domain_of_validity={
            "included": ["measured directions sharing one fixed object, receiver, and model basin"],
            "excluded": ["new causal schedules", "changed representations", "unmeasured directions"],
            "verdict_scope": "FORMULATION(current HPAC same-basin directions)",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"objective_deltas": "score-formula units on one declared axis"},
        units_out={"minimum_delta_s": "score-formula units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: anchor.residual},
        last_calibration_utc="2026-08-24T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("DDM costate organ round 10", "DCC1 successor ranking"),
        canonical_producers=(HC1, "ddm_lv3_recursive_leverage_20260901"),
        provenance=provenance,
    )


def build_byte_distortion_cross_intersection_count_v1() -> CanonicalEquation:
    provenance = _prov(X012, "append a body only after both complete archive bytes and matched realized distortion exist")
    byte_ok = [True, True, False, False]
    distortion_ok = [False, False, True, True]
    count = eval_byte_distortion_cross_intersection_count(
        {"byte_feasible": byte_ok, "distortion_feasible": distortion_ok}
    )
    anchor = _anchor(
        anchor_id="x012_four_body_cross_empty_20260901",
        source=X012,
        inputs={"body_count": 4, "byte_feasible": byte_ok, "distortion_feasible": distortion_ok},
        predicted={"intersection_count": 0},
        empirical={"intersection_count": count, "closest_complete_archive_bytes": 147_327, "gate_bytes": 137_985},
        residual=0.0,
        method="bounded four-body receipt census with complete-archive and matched-distortion predicates kept separate",
        provenance=provenance,
    )
    return CanonicalEquation(
        equation_id=CROSS_ID,
        name="Byte-distortion Cross intersection count",
        one_line_summary="Count only bodies that simultaneously satisfy a complete-archive byte predicate and a matched realized-distortion predicate.",
        latex_form=r"N_\times=\sum_i\mathbf1[B_i\le B_{max}]\mathbf1[D_i\le D_{max}]",
        python_callable_module_path="tac.canonical_equations.evaluators:eval_byte_distortion_cross_intersection_count",
        domain_of_validity={
            "included": ["complete receiver-closed bodies with matched byte and distortion predicates"],
            "excluded": ["payload-only byte counts", "stubs", "unmeasured distortion"],
            "verdict_scope": "INSTANCE(the four measured Cross bodies)",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"byte_feasible": "boolean sequence", "distortion_feasible": "boolean sequence"},
        units_out={"intersection_count": "bodies"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: 0.0},
        last_calibration_utc="2026-09-01T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("DCC1 successor ranking", "current sub-0.12 crossing ledger"),
        canonical_producers=(NI1R, X012, "ddm_lv3_recursive_leverage_20260901"),
        provenance=provenance,
    )


def build_roundtrip_token_to_argmax_affine_v1() -> CanonicalEquation:
    provenance = _prov(LBX1, "refit only from additional matched-PYAV token/argmax pairs on the same receiver")
    predicted = eval_roundtrip_token_to_argmax_affine(
        {"intercept_argmax_errors": 17_241, "marginal_argmax_errors_per_token_error": 1.1435, "token_errors": 0}
    )
    anchor = _anchor(
        anchor_id="gf1_lbx1_matched_pyav_affine_n2_20260831",
        source=LBX1,
        inputs={"matched_points": 2, "intercept_argmax_errors": 17_241, "marginal_argmax_errors_per_token_error": 1.1435, "token_errors": 0},
        predicted={"argmax_errors": predicted},
        empirical={"argmax_errors": 17_241, "through_origin_ratio_1_157_transferable": False, "current_rate_only_ceiling_bytes": 140_477},
        residual=0.0,
        method="two-point matched-PYAV affine fit plus exact score-box ceiling derivation",
        provenance=provenance,
    )
    return CanonicalEquation(
        equation_id=AFFINE_ID,
        name="Round-trip token-to-argmax affine transfer",
        one_line_summary="Matched PYAV transfer has a 17,241-error receiver floor and marginal slope 1.1435; a through-origin 1.157 multiplier is invalid at LB1.",
        latex_form=r"E_{argmax}\approx 17241+1.1435E_{token}",
        python_callable_module_path="tac.canonical_equations.evaluators:eval_roundtrip_token_to_argmax_affine",
        domain_of_validity={
            "included": ["the n=2 matched-PYAV LB1/GF1 receiver cells"],
            "excluded": ["through-origin extrapolation", "new renderers", "promotion of the 140477-byte derived ceiling to an archive measurement"],
            "verdict_scope": "FORMULATION(current matched-PYAV affine transfer)",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"intercept_argmax_errors": "pixels", "marginal_argmax_errors_per_token_error": "pixels/token", "token_errors": "tokens"},
        units_out={"argmax_errors": "pixels"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: 0.0},
        last_calibration_utc="2026-08-31T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("current rate-corner screen", "DCC1 successor ranking"),
        canonical_producers=(GF1, LBX1, "ddm_lv3_recursive_leverage_20260901"),
        provenance=provenance,
    )


def build_field_change_bhw_decomposition_v1() -> CanonicalEquation:
    provenance = _prov(FCD1, "recalibrate only on a new changed-field population with exact token, coding-field, and GT labels")
    counts = eval_field_change_bhw_decomposition(
        {"before_labels": [0, 1, 2], "after_labels": [1, 2, 3], "ground_truth_labels": [1, 1, 4]}
    )
    anchor = _anchor(
        anchor_id="fcd1_full_population_bhw_and_real_coder_union_20260829",
        source=FCD1,
        inputs={"population_positions": 117_964_800, "changed_positions": 227_671, "classification_smoke": counts},
        predicted={"benefit_only_union_may_open_rate": True},
        empirical={"benefit": 5_268, "harm": 221_862, "wash": 541, "selected_union_bhw": [5_268, 0, 0], "archive_before_bytes": 180_192, "archive_after_bytes": 176_436, "delta_bytes": -3_756, "realized_scorer_status": "NOT_MEASURED"},
        residual=0.0,
        method="full-population exact three-field label join followed by retained real joint-coder union encode",
        provenance=provenance,
    )
    return CanonicalEquation(
        equation_id=BHW_ID,
        name="Field-change benefit/harm/wash decomposition",
        one_line_summary="Classify every proposed changed label against ground truth before coding; FCD1's B-only union saved 3,756 bytes but has no realized scorer claim.",
        latex_form=r"B=[x\ne y^*\land x'=y^*],\ H=[x=y^*\land x'\ne y^*],\ W=1-B-H",
        python_callable_module_path="tac.canonical_equations.evaluators:eval_field_change_bhw_decomposition",
        domain_of_validity={
            "included": ["changed token/coding-field positions with exact GT labels"],
            "excluded": ["inference that token-label benefit survives renderer R or PoseNet"],
            "verdict_scope": "FORMULATION(label attribution); FCD1 byte anchor is INSTANCE",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"before_labels": "class ids", "after_labels": "class ids", "ground_truth_labels": "class ids"},
        units_out={"benefit_harm_wash": "counts"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: 0.0},
        last_calibration_utc="2026-08-29T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("fixed-field causal G/M schedule", "joint scorer admission after rate screen"),
        canonical_producers=(FCD1, "ddm_lv3_recursive_leverage_20260901"),
        provenance=provenance,
    )


def build_context_model_reorder_savings_v1() -> CanonicalEquation:
    provenance = _prov(GF1, "append paired order races only when coder class and decoded object are pinned")
    generic = eval_context_model_reorder_savings(
        {"has_context_model": False, "generic_coder_savings_bytes": 70_552, "context_model_savings_bytes": 0}
    )
    context = eval_context_model_reorder_savings(
        {"has_context_model": True, "generic_coder_savings_bytes": 70_552, "context_model_savings_bytes": 0}
    )
    anchor = _anchor(
        anchor_id="gf1_generic_order_win_rr9_context_order_zero_20260830",
        source=GF1,
        inputs={"generic_frame_raster_bytes": 456_000, "generic_tile16_time_bytes": 385_448, "trained_context_stream_bytes": 113_777},
        predicted={"generic_savings_bytes": generic, "context_model_savings_bytes": context},
        empirical={"generic_savings_bytes": 70_552, "generic_relative_savings": 0.15471929824561403, "context_model_savings_bytes": 0},
        residual=0.0,
        method="real order race on generic LZ residual plus source-joined RR9 trained-context control",
        provenance=provenance,
    )
    return CanonicalEquation(
        equation_id=REORDER_ID,
        name="Reorder savings conditional on coder context model",
        one_line_summary="Order races can pay for generic match coders, while the measured trained-context stream is order-invariant at zero bytes saved.",
        latex_form=r"\Delta B_{reorder}>0\ \mathrm{only\ for\ order\ sensitive\ coder};\quad \Delta B_{context}=0",
        python_callable_module_path="tac.canonical_equations.evaluators:eval_context_model_reorder_savings",
        domain_of_validity={
            "included": ["paired lossless order races on an unchanged decoded object"],
            "excluded": ["transfer of generic-coder savings to a trained context model"],
            "verdict_scope": "FORMULATION(coder-class conditional reorder value)",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"savings": "bytes", "has_context_model": "bool"},
        units_out={"admissible_reorder_savings": "bytes"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: 0.0},
        last_calibration_utc="2026-08-30T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("causal schedule coder race", "generic residual order race"),
        canonical_producers=(GF1, ".omx/research/ddm_rr9_reorder_refit_20260824.md", "ddm_lv3_recursive_leverage_20260901"),
        provenance=provenance,
    )


def build_generator_form_fit_error_entanglement_v1() -> CanonicalEquation:
    provenance = _prov(GF1, "reopen only with a different generator formulation that jointly reports exact bytes and fit error")
    anchor = _anchor(
        anchor_id="gf1_generator_ratio_inseparable_from_fit_error_20260830",
        source=GF1,
        inputs={"generator_bytes": 47_603, "fit_error_fraction": 0.0112324, "correction_count": 1_325_033},
        predicted={"transferable_as_lossless_credit": False},
        empirical={"reported_byte_ratio": 2.178, "fit_error_fraction": 0.0112324, "transferable_as_lossless_credit": False, "exact_correction_stream_bytes": 385_448},
        residual=0.0,
        method="same generator fit to two distinct targets plus retained exact correction-stream coder race",
        provenance=provenance,
    )
    return CanonicalEquation(
        equation_id=GENERATOR_ID,
        name="Generator-form byte ratio and fit-error entanglement",
        one_line_summary="The 2.178x generator ratio is real only together with about 1.12% fit error; it is not transferable as a lossless form credit.",
        latex_form=r"\rho_B=B_{ref}/B_G\quad\mathrm{reported\ with}\quad e_{fit};\ e_{fit}>0\Rightarrow\rho_B\ \mathrm{not\ lossless\ credit}",
        python_callable_module_path="tac.canonical_equations.evaluators:eval_generator_form_fit_error_entanglement",
        domain_of_validity={
            "included": ["the GF1/HG1 analytic four-stream generator and exact retained correction price"],
            "excluded": ["claim that the generator form can be inherited without its fit error", "other generator formulations"],
            "verdict_scope": "FORMULATION(the tested four-stream analytic generator)",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"reference_bytes": "bytes", "generator_bytes": "bytes", "fit_error_fraction": "fraction"},
        units_out={"byte_ratio": "dimensionless", "fit_error_fraction": "fraction", "transferable_as_lossless_credit": "bool"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: anchor.residual},
        last_calibration_utc="2026-08-30T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("changed-object generator screens", "DCC1 successor ranking"),
        canonical_producers=(GF1, "ddm_lv3_recursive_leverage_20260901"),
        provenance=provenance,
    )


def build_decoder_derivable_ideal_savings_ceiling_v1() -> CanonicalEquation:
    provenance = _prov(DDS1, "recompute when sample fraction, conditional context, or full-population demand changes")
    full_tuple = eval_decoder_derivable_ideal_savings_ceiling(
        {"sampled_gain_bits": 981.498, "sampled_fraction": 0.2}
    )
    m_only = eval_decoder_derivable_ideal_savings_ceiling(
        {"sampled_gain_bits": 3.322, "sampled_fraction": 0.2}
    )
    anchor = _anchor(
        anchor_id="dds1_decoder_derivable_ceiling_readjudication_20260901",
        source=DDS1,
        inputs={"sampled_fraction": 0.2, "full_tuple_gain_bits": 981.498, "m_only_gain_bits": 3.322, "door_r_demand_bytes": 42_016},
        predicted={"full_tuple_ideal_ceiling_bytes": full_tuple, "m_only_ideal_ceiling_bytes": m_only},
        empirical={"full_tuple_ideal_ceiling_bytes_rounded": 613, "m_only_ideal_ceiling_bytes": m_only, "m_only_disposition": "CLOSED-BY-CEILING"},
        residual=abs(full_tuple - 613.0),
        method="scale exact n120 conditional-codelength gains to the full population; use only as an optimistic refusal ceiling",
        provenance=provenance,
        inspected=True,
    )
    return CanonicalEquation(
        equation_id=CEILING_ID,
        name="Decoder-derivable ideal savings ceiling",
        one_line_summary="Scale a sampled decoder-realizable conditional-codelength gain into an optimistic byte ceiling usable to refuse work, never as a byte claim.",
        latex_form=r"B_{ideal,max}=G_{sample,bits}/(8f_{sample})",
        python_callable_module_path="tac.canonical_equations.evaluators:eval_decoder_derivable_ideal_savings_ceiling",
        domain_of_validity={
            "included": ["ceiling-first refusal where the context is decoder-realizable and the sample fraction is declared"],
            "excluded": ["physical byte claims", "oracle-only wrong-site context", "admission based on a screen estimate"],
            "verdict_scope": "FORMULATION(DDS1 full tuple and M-only rider)",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"sampled_gain_bits": "bits", "sampled_fraction": "fraction"},
        units_out={"ideal_savings_ceiling": "bytes"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={anchor.anchor_id: anchor.residual},
        last_calibration_utc="2026-09-01T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("SCMDL admission screen", "DCC1 successor ranking"),
        canonical_producers=(DDS1, "ddm_lv3_recursive_leverage_20260901"),
        provenance=provenance,
    )


ALL_LV3_CURRENT_ARC_BUILDERS: tuple[Callable[[], CanonicalEquation], ...] = (
    build_same_basin_sharp_optimum_v1,
    build_byte_distortion_cross_intersection_count_v1,
    build_roundtrip_token_to_argmax_affine_v1,
    build_field_change_bhw_decomposition_v1,
    build_context_model_reorder_savings_v1,
    build_generator_form_fit_error_entanglement_v1,
    build_decoder_derivable_ideal_savings_ceiling_v1,
)


def populate_lv3_current_arc_laws(*, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None) -> tuple[CanonicalEquation, ...]:
    """Append the seven non-duplicate current-arc laws to the canonical registry."""

    from tac.canonical_equations.registry import register_canonical_equation

    equations = tuple(builder() for builder in ALL_LV3_CURRENT_ARC_BUILDERS)
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes="DDM LV3 current-arc recursive-leverage law registration; advisory, score_claim=false",
        )
    return equations


__all__ = [
    "AFFINE_ID",
    "ALL_LV3_CURRENT_ARC_BUILDERS",
    "BHW_ID",
    "CEILING_ID",
    "CROSS_ID",
    "GENERATOR_ID",
    "REORDER_ID",
    "SHARP_ID",
    "build_byte_distortion_cross_intersection_count_v1",
    "build_context_model_reorder_savings_v1",
    "build_decoder_derivable_ideal_savings_ceiling_v1",
    "build_field_change_bhw_decomposition_v1",
    "build_generator_form_fit_error_entanglement_v1",
    "build_roundtrip_token_to_argmax_affine_v1",
    "build_same_basin_sharp_optimum_v1",
    "populate_lv3_current_arc_laws",
]
