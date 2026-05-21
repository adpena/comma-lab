# SPDX-License-Identifier: MIT
"""Canonical equation: procedural predictor plus residual correction savings.

This is the sister equation to
``procedural_codebook_from_seed_compression_savings_v1`` for contexts that
Catalog #359 deliberately excludes from equation #26.

Equation #26 predicts direct replacement:

    archive bytes = K_seed instead of N_original

Residual-hybrid rescue paths are different:

    archive bytes = K_predictor + R_residual + H_envelope

The residual stream is charged. This module makes that byte accounting explicit
so future magic-codec residual experiments cannot reuse the replacement-savings
formula and accidentally treat residual bytes as free.
"""
from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    CanonicalEquation,
    DomainOfValidityViolation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.canonical_equations.procedural_codebook_savings import (
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    is_residual_hybrid_context,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)


EQUATION_ID = "procedural_predictor_plus_residual_correction_savings_v1"


def _require_nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def validate_residual_hybrid_context(
    context: str | None,
    *,
    raise_on_invalid: bool = True,
) -> bool:
    """Validate that ``context`` belongs to the residual-hybrid equation.

    This is the positive-domain counterpart to Catalog #359's equation #26
    refusal gate. If a context does not match the canonical residual-hybrid
    pattern, callers should route it to another equation instead of using this
    one by habit.
    """

    if is_residual_hybrid_context(context):
        return True
    if not raise_on_invalid:
        return False
    raise DomainOfValidityViolation(
        f"context={context!r} is not a residual-hybrid context for {EQUATION_ID}; "
        "use this equation only for predictor-plus-residual archive byte "
        "accounting, and use procedural_codebook_from_seed_compression_savings_v1 "
        "for direct codebook replacement contexts."
    )


def predict_procedural_predictor_plus_residual_correction_savings(
    *,
    original_payload_bytes: int,
    predictor_seed_or_code_bytes: int,
    residual_stream_bytes: int,
    container_overhead_bytes: int = 0,
    context: str | None,
) -> dict[str, Any]:
    """Compute rate-axis delta for predictor-plus-residual archive accounting.

    Negative ``predicted_delta_s_rate_only`` is a rate win. Positive is a rate
    regression. This function intentionally predicts only the contest rate term;
    it makes no SegNet/PoseNet distortion claim.
    """

    validate_residual_hybrid_context(context)
    original = _require_nonnegative_int(original_payload_bytes, "original_payload_bytes")
    predictor = _require_nonnegative_int(
        predictor_seed_or_code_bytes,
        "predictor_seed_or_code_bytes",
    )
    residual = _require_nonnegative_int(residual_stream_bytes, "residual_stream_bytes")
    overhead = _require_nonnegative_int(container_overhead_bytes, "container_overhead_bytes")
    replacement_total = predictor + residual + overhead
    delta_bytes = replacement_total - original
    predicted_delta_s = (
        CANONICAL_RATE_MULTIPLIER * float(delta_bytes) / CANONICAL_RATE_DENOM_BYTES
    )
    if delta_bytes < 0:
        verdict = "RATE_WIN"
    elif delta_bytes > 0:
        verdict = "RATE_REGRESSION"
    else:
        verdict = "RATE_NEUTRAL"
    return {
        "equation_id": EQUATION_ID,
        "context": context,
        "original_payload_bytes": original,
        "predictor_seed_or_code_bytes": predictor,
        "residual_stream_bytes": residual,
        "container_overhead_bytes": overhead,
        "replacement_total_bytes": replacement_total,
        "delta_bytes_replacement_minus_original": delta_bytes,
        "bytes_saved": -delta_bytes,
        "predicted_delta_s_rate_only": predicted_delta_s,
        "prediction_scope": "rate_axis_only_no_scorer_distortion_claim",
        "verdict": verdict,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _anchor_pair_1_dwt_detail_residual() -> EmpiricalAnchor:
    predicted = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=131_779,
        predictor_seed_or_code_bytes=96,
        residual_stream_bytes=186_958,
        container_overhead_bytes=0,
        context="magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
    )
    empirical_delta = 0.036805353633828024
    return EmpiricalAnchor(
        anchor_id="pair_1_dwt_detail_dense_streams_residual_rate_accounting_20260520",
        measurement_utc="2026-05-20T23:47:04Z",
        inputs={
            "in_domain_context": "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
            "baseline_bytes": 131_779,
            "procedural_predictor_bytes": 96,
            "procedural_plus_residual_bytes": 187_054,
            "residual_stream_plus_envelope_bytes": 186_958,
            "historical_misapplied_equation": "procedural_codebook_from_seed_compression_savings_v1",
        },
        predicted_output=predicted,
        empirical_output={
            "empirical_delta_s": empirical_delta,
            "empirical_delta_bytes": 55_275,
            "observed_verdict": "CARGO_CULTED_RATE_REGRESSION",
        },
        residual=abs(empirical_delta - float(predicted["predicted_delta_s_rate_only"])),
        source_artifact="experiments/results/magic_codec_dense_streams_dwt_residual_smoke_20260520T234704Z/smoke_result.json",
        measurement_method="local_byte_accounting_smoke_config_c_minus_config_a",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="experiments/results/magic_codec_dense_streams_dwt_residual_smoke_20260520T234704Z/smoke_result.json",
            reactivation_criteria="procedural_predictor_plus_residual_rate_equation_pair_1_anchor",
            measurement_axis="[byte-budget local smoke only]",
            hardware_substrate="darwin_arm64_m5_max_macos_cpu_advisory",
        ),
    )


def _anchor_pair_2_fec6_null_byte_srl1_residual() -> EmpiricalAnchor:
    predicted = predict_procedural_predictor_plus_residual_correction_savings(
        original_payload_bytes=16_292,
        predictor_seed_or_code_bytes=32,
        residual_stream_bytes=97_441,
        container_overhead_bytes=0,
        context="sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes",
    )
    empirical_delta = 0.05405509567341099
    return EmpiricalAnchor(
        anchor_id="pair_2_fec6_null_byte_srl1_residual_rate_accounting_20260521",
        measurement_utc="2026-05-21T00:21:20Z",
        inputs={
            "in_domain_context": "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes",
            "baseline_in_place_charged_bytes": 16_292,
            "procedural_predictor_bytes": 32,
            "procedural_plus_srl1_residual_bytes": 97_473,
            "srl1_residual_plus_envelope_bytes": 97_441,
            "null_byte_count": 16_292,
            "historical_misapplied_equation": "procedural_codebook_from_seed_compression_savings_v1",
        },
        predicted_output=predicted,
        empirical_output={
            "empirical_delta_s": empirical_delta,
            "empirical_delta_bytes": 81_181,
            "observed_verdict": "CARGO_CULTED_RATE_REGRESSION",
        },
        residual=abs(empirical_delta - float(predicted["predicted_delta_s_rate_only"])),
        source_artifact="experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z/smoke_result.json",
        measurement_method="local_byte_accounting_smoke_config_c_minus_config_a",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z/smoke_result.json",
            reactivation_criteria="procedural_predictor_plus_residual_rate_equation_pair_2_anchor",
            measurement_axis="[byte-budget local smoke only]",
            hardware_substrate="darwin_arm64_m5_max_macos_cpu_advisory",
        ),
    )


def build_procedural_predictor_plus_residual_correction_savings_v1() -> CanonicalEquation:
    """Build the residual-hybrid byte-accounting equation with two anchors."""

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Procedural predictor plus residual correction savings",
        one_line_summary=(
            "Rate-only ΔS for residual-hybrid archives equals 25*(K+R+H-N)/37,545,489; residual bytes are charged."
        ),
        latex_form=(
            r"\Delta S_{\text{residual}} = "
            r"\frac{25 \cdot (K_{\text{predictor}} + R_{\text{residual}} "
            r"+ H_{\text{envelope}} - N_{\text{original}})}{37{,}545{,}489}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.procedural_predictor_residual_savings:"
            "predict_procedural_predictor_plus_residual_correction_savings"
        ),
        domain_of_validity={
            "contexts": [
                "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
                "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes",
                "future_procedural_predictor_plus_residual_contexts_matching_catalog_359_patterns",
            ],
            "requires_residual_hybrid_context": True,
            "prediction_scope": "rate_axis_only_no_scorer_distortion_claim",
            "canonical_rate_denom_bytes": CANONICAL_RATE_DENOM_BYTES,
            "canonical_rate_multiplier": CANONICAL_RATE_MULTIPLIER,
            "non_claim_authority": {
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
            },
        },
        units_in={
            "original_payload_bytes": "int_archive_member_bytes_count",
            "predictor_seed_or_code_bytes": "int_archive_member_bytes_count",
            "residual_stream_bytes": "int_archive_member_bytes_count",
            "container_overhead_bytes": "int_archive_member_bytes_count",
        },
        units_out={
            "delta_bytes_replacement_minus_original": "int_archive_member_bytes_count",
            "predicted_delta_s_rate_only": "float_score_axis_delta_lower_is_better",
        },
        empirical_anchors=(
            _anchor_pair_1_dwt_detail_residual(),
            _anchor_pair_2_fec6_null_byte_srl1_residual(),
        ),
        predicted_vs_empirical_residual={
            "pair_1_dwt_detail_dense_streams_residual": 0.0,
            "pair_2_fec6_null_byte_srl1_residual": 0.0,
        },
        last_calibration_utc="2026-05-21T01:05:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.canonical_equations.procedural_codebook_savings.refuse_residual_hybrid_context_misapplication",
            "future_magic_codec_residual_hybrid_smokes",
            "tac.optimization.scorer_response_dataset",
        ),
        canonical_producers=(
            "tools/run_magic_codec_dense_streams_dwt_residual_smoke.py",
            "tools/run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="procedural_predictor_plus_residual_correction_savings.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="not_applicable_rate_formula",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "build_procedural_predictor_plus_residual_correction_savings_v1",
    "predict_procedural_predictor_plus_residual_correction_savings",
    "validate_residual_hybrid_context",
]
