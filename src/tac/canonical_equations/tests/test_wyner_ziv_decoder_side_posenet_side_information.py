# SPDX-License-Identifier: MIT
"""Tests for the Wyner-Ziv decoder-side PoseNet side-information canonical equation.

Per operator task #1496 Wave N+36 routing + CLAUDE.md "Canonical equations
+ models registry" non-negotiable + Catalog #344 sister discipline.
"""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1,
    predict_wyner_ziv_posenet_side_info_savings,
    query_equations,
)
from tac.canonical_equations.equation import (
    INFERRED_FROM_DOMAIN_LITERATURE,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
from tac.canonical_equations.wyner_ziv_decoder_side_posenet_side_information import (
    EQUATION_ID,
)
from tac.provenance.contract import Provenance


# ---------------------------------------------------------------------------
# Canonical equation builder invariants
# ---------------------------------------------------------------------------


def test_canonical_equation_builder_returns_well_formed_equation() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    assert isinstance(eq, CanonicalEquation)
    assert eq.equation_id == EQUATION_ID
    assert "wyner_ziv_decoder_side_posenet_side_information" in eq.equation_id
    assert eq.equation_id.endswith("_v1")


def test_canonical_equation_has_required_metadata_fields() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    assert eq.name.strip()
    assert eq.one_line_summary.strip()
    assert "Wyner-Ziv" in eq.one_line_summary or "R(D|Y)" in eq.one_line_summary
    assert len(eq.one_line_summary) <= 200
    assert eq.latex_form.strip()
    assert r"R(D|Y)" in eq.latex_form
    assert "PoseNet" in eq.latex_form or r"\mathrm{PoseNet}" in eq.latex_form


def test_canonical_equation_callable_path_resolvable() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    assert eq.python_callable_module_path == (
        "tac.canonical_equations.wyner_ziv_decoder_side_posenet_side_information:"
        "predict_wyner_ziv_posenet_side_info_savings"
    )


def test_canonical_equation_domain_of_validity_has_in_and_excluded() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    dom = dict(eq.domain_of_validity)
    assert "in_domain" in dom
    assert "excluded" in dom
    assert isinstance(dom["in_domain"], list)
    assert isinstance(dom["excluded"], list)
    assert len(dom["in_domain"]) >= 1
    assert len(dom["excluded"]) >= 3
    # Each excluded entry must carry a context_id + rationale.
    for entry in dom["excluded"]:
        assert "context_id" in entry
        assert "rationale" in entry


def test_canonical_equation_excludes_posenet_as_source_degenerate_case() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    excluded_ids = {e["context_id"] for e in eq.domain_of_validity["excluded"]}
    assert "posenet_as_source_degenerate" in excluded_ids
    assert "non_video_signals" in excluded_ids
    assert "non_decoder_reproducible_substrates" in excluded_ids
    # Catalog #359 sister discipline: REPLACEMENT-savings equations don't
    # apply to RESIDUAL-CORRECTION-stacking contexts.
    assert "residual_hybrid_contexts_per_catalog_359" in excluded_ids


def test_canonical_equation_first_anchor_cites_z8_m6() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    assert len(eq.empirical_anchors) == 1
    anchor = eq.empirical_anchors[0]
    assert isinstance(anchor, EmpiricalAnchor)
    assert "z8_m6" in anchor.anchor_id
    assert "5d5634dd3" in str(anchor.inputs.get("z8_m6_commit_sha", ""))
    # Z8 M6 landing memo cited.
    assert "z8_m6" in anchor.source_artifact
    # Per Catalog #363 the verification status MUST be one of the canonical 4
    # values; this anchor cites the Z8 M6 empirical landing so it is
    # VERIFIED_VIA_EMPIRICAL_ANCHOR per the 4-value taxonomy.
    assert anchor.empirical_verification_status == VERIFIED_VIA_EMPIRICAL_ANCHOR


def test_canonical_equation_first_anchor_carries_canonical_provenance() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    anchor = eq.empirical_anchors[0]
    assert isinstance(anchor.provenance, Provenance)
    # Per Catalog #192 macOS-local-CPU advisory IS NEVER promotable.
    assert anchor.provenance.promotion_eligible is False
    assert anchor.provenance.score_claim_valid is False
    # Anchor axis tagged macOS-CPU advisory.
    assert "macOS-CPU advisory" in anchor.provenance.measurement_axis


def test_canonical_equation_predicts_within_calibration_tolerance() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    # The Z8 M6 paradigm-anchor residual: |predicted - empirical| / max.
    # Predicted savings = 523 * 0.736 = 385 bytes; empirical = 523 - 138 = 385.
    # Residual should be 0 (predicted matches empirical exactly at the
    # canonical synthetic-Gaussian high-correlation reference point).
    residual = eq.predicted_vs_empirical_residual[
        "synthetic_gaussian_z8_m6_paradigm_anchor"
    ]
    # Canonical 2x tolerance per is_well_calibrated; we expect much tighter.
    assert residual < 2.0
    assert residual >= 0


def test_canonical_equation_is_well_calibrated_at_landing() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    assert eq.is_well_calibrated is True


def test_canonical_equation_units_declared() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    assert "source_bytes_unconditional" in eq.units_in
    assert "mutual_information_estimate_bits" in eq.units_in
    assert "bytes_saved_predicted" in eq.units_out
    assert "predicted_delta_s_rate_axis" in eq.units_out


def test_canonical_equation_consumers_include_cathedral_sister() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    consumers = list(eq.canonical_consumers)
    assert (
        "tac.cathedral_consumers.wyner_ziv_posenet_side_information_consumer"
        in consumers
    )
    assert "tac.cathedral_consumers.canonical_equation_lookup_consumer" in consumers
    assert "tools.cathedral_autopilot_autonomous_loop" in consumers


def test_canonical_equation_producers_include_z8_m6_and_upstream_posenet() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    producers = list(eq.canonical_producers)
    assert "upstream.modules.PoseNet" in producers
    assert any("z8_hierarchical_predictive_coding" in p for p in producers)
    assert any("predict_wyner_ziv_posenet_side_info_savings" in p for p in producers)


def test_canonical_equation_recalibration_trigger_is_on_new_anchors() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    # Per Catalog #371 auto-recalibrator: triggers when >=3 new in-domain
    # anchors land (the canonical default for new equations).
    assert eq.next_recalibration_trigger == RECALIBRATE_ON_NEW_ANCHORS


def test_canonical_equation_provenance_is_predicted_grade() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    assert isinstance(eq.provenance, Provenance)
    # Equation-level Provenance is PREDICTED per Catalog #323 (the equation
    # itself is a model; per-anchor Provenance carries the empirical grade).
    assert eq.provenance.measurement_axis == "[predicted]"
    assert eq.provenance.promotion_eligible is False


# ---------------------------------------------------------------------------
# Round-trip serialization (Catalog #128/#131 sister fcntl-locked JSONL)
# ---------------------------------------------------------------------------


def test_canonical_equation_round_trips_as_dict() -> None:
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    payload = eq.to_dict()
    assert payload["equation_id"] == EQUATION_ID
    assert payload["schema_version"] == eq.schema_version
    assert len(payload["empirical_anchors"]) == 1


# ---------------------------------------------------------------------------
# predict_wyner_ziv_posenet_side_info_savings helper invariants
# ---------------------------------------------------------------------------


def test_predict_helper_returns_dict_with_canonical_fields() -> None:
    result = predict_wyner_ziv_posenet_side_info_savings(
        source_bytes_unconditional=1000,
        mutual_information_estimate_bits=2.0,
        source_distortion_target=0.1,
    )
    assert result["equation_id"] == EQUATION_ID
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["axis_tag"] == "[predicted]"
    assert "Wyner-Ziv 1976" in result["rationale"]


def test_predict_helper_zero_correlation_zero_savings() -> None:
    result = predict_wyner_ziv_posenet_side_info_savings(
        source_bytes_unconditional=1000,
        mutual_information_estimate_bits=0.0,
        source_distortion_target=0.1,
    )
    assert result["bytes_saved_predicted"] == 0
    assert result["savings_ratio_predicted"] == 0.0
    assert result["predicted_delta_s_rate_axis"] == 0.0


def test_predict_helper_max_correlation_higher_savings() -> None:
    low = predict_wyner_ziv_posenet_side_info_savings(
        source_bytes_unconditional=1000,
        mutual_information_estimate_bits=2.0,
        source_distortion_target=0.1,
        side_info_correlation_proxy=0.1,
    )
    high = predict_wyner_ziv_posenet_side_info_savings(
        source_bytes_unconditional=1000,
        mutual_information_estimate_bits=2.0,
        source_distortion_target=0.1,
        side_info_correlation_proxy=0.9,
    )
    assert high["bytes_saved_predicted"] > low["bytes_saved_predicted"]


def test_predict_helper_rate_axis_delta_uses_canonical_contest_constants() -> None:
    # Per CLAUDE.md canonical rate term: -25 * bytes_saved / 37545489
    result = predict_wyner_ziv_posenet_side_info_savings(
        source_bytes_unconditional=37_545_489,
        mutual_information_estimate_bits=48.0,  # equals side_info_dim*8 max bits
        source_distortion_target=0.1,
        side_info_dim=6,
        side_info_correlation_proxy=1.0,
    )
    # At maximal correlation + maximal MI, savings ratio = 1.0 so bytes_saved
    # = source bytes; rate-axis ΔS = -25.0 exactly.
    assert math.isclose(result["predicted_delta_s_rate_axis"], -25.0, abs_tol=1e-3)


def test_predict_helper_rejects_negative_bytes() -> None:
    with pytest.raises(ValueError, match="source_bytes_unconditional"):
        predict_wyner_ziv_posenet_side_info_savings(
            source_bytes_unconditional=-1,
            mutual_information_estimate_bits=2.0,
            source_distortion_target=0.1,
        )


def test_predict_helper_rejects_negative_mutual_info() -> None:
    with pytest.raises(ValueError, match="mutual_information_estimate_bits"):
        predict_wyner_ziv_posenet_side_info_savings(
            source_bytes_unconditional=100,
            mutual_information_estimate_bits=-0.5,
            source_distortion_target=0.1,
        )


def test_predict_helper_rejects_out_of_range_correlation() -> None:
    with pytest.raises(ValueError, match="side_info_correlation_proxy"):
        predict_wyner_ziv_posenet_side_info_savings(
            source_bytes_unconditional=100,
            mutual_information_estimate_bits=2.0,
            source_distortion_target=0.1,
            side_info_correlation_proxy=1.5,
        )


def test_predict_helper_zero_bytes_input_returns_zero_savings() -> None:
    result = predict_wyner_ziv_posenet_side_info_savings(
        source_bytes_unconditional=0,
        mutual_information_estimate_bits=2.0,
        source_distortion_target=0.1,
    )
    assert result["bytes_saved_predicted"] == 0
    assert result["bytes_after_wyner_ziv_predicted"] == 0


# ---------------------------------------------------------------------------
# Catalog #344 sister discipline regression guards
# ---------------------------------------------------------------------------


def test_equation_is_distinct_from_sister_wyner_ziv_equations() -> None:
    """Verify the new equation is registered with a distinct equation_id."""
    eq = build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1()
    # The 6 sister Wyner-Ziv equations are pinned by name; ensure no collision.
    sister_ids = {
        "wyner_ziv_decoder_side_information_conditional_entropy_savings_v1",
        "wyner_ziv_pipeline_stage_codec_decoder_side_canonical_y_savings_v1",
        "wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1",
        "wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1",
        "wyner_ziv_y_derivable_3_surface_convergence_density_ceiling_v1",
        "wyner_ziv_decoder_side_information_class_shift_refined_predicted_score_delta_v1",
    }
    assert eq.equation_id not in sister_ids


def test_equation_id_matches_canonical_pattern() -> None:
    """Per equation.py _EQUATION_ID_RE: snake_case + trailing _vN."""
    import re

    canonical_pattern = re.compile(r"^[a-z][a-z0-9_]*_v\d+$")
    assert canonical_pattern.match(EQUATION_ID)
