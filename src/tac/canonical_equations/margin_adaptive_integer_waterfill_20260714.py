# SPDX-License-Identifier: MIT
"""Canonical law for exact-integer profile selection at an argmax boundary.

This is an analytic, registration-inert equation.  The pending M5-Max n600
receipt is the first empirical anchor; this module deliberately does not
pre-register or fabricate it.
"""

from __future__ import annotations

from tac.canonical_equations.equation import RECALIBRATE_ON_NEW_ANCHORS, CanonicalEquation
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "margin_adaptive_integer_profile_waterfill_v1"
DERIVATION_UTC = "2026-07-14T11:20:35Z"
MEMO = ".omx/research/margin_adaptive_mixed_precision_DAG_FEED_20260714.md"


def build_margin_adaptive_integer_profile_waterfill_v1() -> CanonicalEquation:
    """Build the finite-ladder KKT/reverse-waterfill admission law."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "attach the exact real-n600 M5-Max receipt only after design-frozen validation, "
            "strict interval-or-frozen-tie coverage, ten-process identity, and positive timing"
        ),
        measurement_axis="[DERIVED analytic law; research-only MEANS; no score authority]",
        hardware_substrate="backend_free_python_reference",
        captured_at_utc=DERIVATION_UTC,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Margin-adaptive exact-integer profile waterfill",
        one_line_summary=(
            "Choose the fastest exact-int profile preserving every winner; separately compute the minimum-bit certifying profile per source pixel."
        ),
        latex_form=(
            r"L_{p,a}^{(k)}=z_{p,a}^{fp32}-e_{p,a}^{(k)},\ "
            r"U_{p,c}^{(k)}=z_{p,c}^{fp32}+e_{p,c}^{(k)},\ "
            r"C_{p,k}=1[L_{p,a}^{(k)}>\max_{c\ne a}U_{p,c}^{(k)}]\lor T_{p,k};\ "
            r"k_{native}^*=\arg\min_{k\in\mathcal K}t_k\ \mathrm{s.t.}\ "
            r"\forall p\in D\cup V:C_{p,k}=1,\ H_k^{(1)}=\cdots=H_k^{(10)},\ t_{fp32}/t_k>1;\ "
            r"k_p^*=\arg\min_{k:C_{p,k}=1}\bar b_k;\ "
            r"A\in\mathrm{int64},\ A=\sum_i q_{x,i}q_{w,i},\ |A|\le2^{63}-1"
        ),
        python_callable_module_path=(
            "tac.local_acceleration.margin_adaptive_mixed_precision:"
            "solve_finite_profile_waterfill"
        ),
        domain_of_validity={
            "research_only": True,
            "domain": (
                "finite frame-independent per-layer precision profiles with per-output-channel "
                "weight scales, native int8/int16/int32 operand storage, dynamic per-layer "
                "activation scales, exact signed-int64 MAC, "
                "corpus-observed classwise logit errors, and a pre-frozen tie rule"
            ),
            "verdict_scope": (
                "n600 SOURCE-CORPUS INSTANCE for the supplied profile ladder; failure is a "
                "FORMULATION-level negative on that granularity, not the mixed-precision family"
            ),
            "authority": "NumPy-fp32/one-thread CPU-Torch winner reference; M5-Max Metal is local candidate only",
            "req_R": (
                "exact real n600 pairs 0..599, design 0..263, second validation 264..599 without "
                "reselection, all-pixel certificate, ten fresh-process digest, and measured timing"
            ),
            "bound_kind": "CORPUS_OBSERVED_PER_PIXEL_ABS_FP32_VS_FIXEDPOINT_LOGIT_ERROR",
            "unseen_input_ibp_claim": False,
            "spatial_waterfill_native_execution_claim": False,
            "distinct_from": (
                "certified_layer_precision_waterfill_v1: generic error-budget allocation",
                "interval_argmax_enclosure_certificate_v1: per-pixel sufficient condition",
                "decode_determinism_integer_arithmetic_v1: exact integer decode discipline",
                "exact_commutative_reduction_reorder_invariance_v1: range-safe reduction theorem",
                "uniform fixed-scale QDQ rung-2: measured NO-GO predecessor, not this exact-int profile formulation",
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
        units_in={
            "reference_logits_z": "logit_units",
            "classwise_error_e": "logit_units",
            "profile_average_bits": "MAC_weighted_bits_per_multiply",
            "profile_latency": "seconds_per_pair",
            "integer_accumulator_A": "signed_integer_units",
        },
        units_out={
            "certificate_C": "boolean_per_pixel_profile",
            "selected_profile_k": "profile_id",
            "selected_average_bits": "MAC_weighted_bits_per_multiply",
            "speedup": "dimensionless_ratio",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=DERIVATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_margin_adaptive_mixed_precision_n600",
            "tac.witness_dsl.margin_adaptive_mixed_precision_20260714",
        ),
        canonical_producers=(
            "tac.local_acceleration.margin_adaptive_mixed_precision",
            "tools.probe_margin_adaptive_mixed_precision_n600",
        ),
        provenance=provenance,
    )


__all__ = ["EQUATION_ID", "build_margin_adaptive_integer_profile_waterfill_v1"]
