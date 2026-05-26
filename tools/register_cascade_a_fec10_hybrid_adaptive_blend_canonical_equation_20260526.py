# SPDX-License-Identifier: MIT
"""One-shot canonical equation registration for Cascade A FEC10 hybrid adaptive-blend savings.

Per operator-approved Cascade A pure-rate-attack spawn (entropy-position cascade exploit catalog
TOP-3 EXPLOIT 2; operator-approval 2026-05-26 verbatim "all are approved + follow up are
approved + pursue other attacks as well + remember all MLX first + individually fractally
optimized") + Catalog #344 operator-decision protocol, registers 1 NEW canonical equation:

  ``cascade_a_fec10_hybrid_adaptive_blend_savings_v1`` — PARADIGM-VALIDATED today via empirical
  236B wire-byte landing on live PR110 600-pair selector stream (-13B vs FEC6 baseline 249B;
  -3B vs FEC8 2nd-order 239B). Adaptive-blend rule with α=2:

    w = ctx2_row_sum / (ctx2_row_sum + α)
    p_blend = w * p_2nd[ctx2_idx] + (1-w) * p_1st[ctx1_idx]

  Decoder-side deterministic; NO flag stream. Sister-disjoint to the originally-prompted
  P11+P13+P15 cascade design (P13 + P15 EMPIRICALLY FALSIFIED per Catalog #307 at 600-symbol
  scale: block-flags +0.5B / brotli +4B).

Registration carries canonical Provenance per Catalog #323 and canonical_producers +
canonical_consumers per Catalog #344 + CLAUDE.md "Subagent coherence-by-default" producer→consumer
audit.

Sister of `markov_context_selector_stream_compression_savings_v1` (FEC8 1st-order parent
equation registered today commit 04f34ea40) + `markov_context_selector_stream_compression_savings_v1`
empirical anchors.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    register_canonical_equation,
)
from tac.provenance import build_provenance_for_predicted


NOW_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
FEC6_ARCHIVE_SHA = "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"


def _build_fec10_hybrid_equation() -> CanonicalEquation:
    """Cascade A FEC10 hybrid adaptive-blend Markov context coder on K=16 selector stream."""
    anchor_provenance = build_provenance_for_predicted(
        model_id="fec10_hybrid_adaptive_blend_v1_alpha_2_static_2nd_order_table",
        inputs_sha256=FEC6_ARCHIVE_SHA,
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_m5_max_macos_cpu_advisory",
        captured_at_utc="2026-05-26T20:55:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="cascade_a_fec10_hybrid_adaptive_blend_alpha_2_live_fec6_stream_20260526",
        measurement_utc="2026-05-26T20:55:00Z",
        inputs={
            "n_pairs": 600,
            "k_palette": 16,
            "alpha": 2,
            "blend_rule": "w = ctx2_row_sum / (ctx2_row_sum + alpha); p_blend = w*p_2nd + (1-w)*p_1st",
            "h_marginal_bits_per_pair": 3.2116,
            "h_first_order_bits_per_pair": 2.9402,
            "h_second_order_true_bits_per_pair": 1.9788,
            "h_blend_bits_per_pair_alpha_2": 3.03,
            "transition_table_kind": "static_hard_coded_zero_transmit",
            "live_stream_archive_sha256": FEC6_ARCHIVE_SHA,
            "in_domain_context": "selector_index_stream_pr101_frame_exploit_600_pairs_k16_palette",
            # Catalog #359 sister discipline: adaptive-blend Markov is NOT residual-hybrid;
            # it operates BEFORE entropy coder via soft mixing of 1st + 2nd-order Markov priors.
        },
        predicted_output={
            "fec10_hybrid_wire_bytes_total": 237,  # predicted from blend bit-rate analysis
            "delta_bytes_vs_fec6": -12,
            "delta_bytes_vs_fec8_2nd_order": -2,
        },
        empirical_output={
            "fec10_hybrid_wire_bytes_total": 236,
            "delta_bytes_vs_fec6": -13,
            "delta_bytes_vs_fec8_2nd_order": -3,
            "delta_rate_contest_units": -8.66e-6,  # = -13 * 25 / 37545489 per CLAUDE.md formula
            "roundtrip_verified": True,
            "encode_time_ms": 0.5,
            "decode_time_ms": 0.7,
        },
        residual=1.0,  # normalized magnitude: empirical wire 236 vs predicted 237 = 1 byte better
        source_artifact=(
            ".omx/research/cascade_a_fec10_hybrid_artifacts_20260526/"
            "cascade_a_fec10_hybrid_empirical.json"
        ),
        measurement_method=(
            "fec10_hybrid_adaptive_blend_alpha_2_encoder_run_on_live_fec6_selector_stream_then_compare_wire_bytes"
        ),
        provenance=anchor_provenance,
    )
    eq_provenance = build_provenance_for_predicted(
        model_id="cascade_a_fec10_hybrid_adaptive_blend_savings_v1",
        inputs_sha256=FEC6_ARCHIVE_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic",
        captured_at_utc=NOW_UTC,
    )
    return CanonicalEquation(
        equation_id="cascade_a_fec10_hybrid_adaptive_blend_savings_v1",
        name="Cascade A FEC10 hybrid adaptive-blend Markov context coder savings",
        one_line_summary=(
            "Predicts wire-byte savings vs FEC8 2nd-order via adaptive 1st/2nd-order "
            "Markov soft-blend (row-sum determines weight; static tables; zero transmit)"
        ),
        latex_form=(
            r"w(c_2) = \frac{\text{row\_sum}_2(c_2)}{\text{row\_sum}_2(c_2) + \alpha}; \quad "
            r"p_{\text{blend}}(x \mid c_2) = w(c_2) \cdot p_2(x \mid c_2) + (1 - w(c_2)) \cdot p_1(x \mid c_1); \quad "
            r"\Delta B = \frac{N}{8} \cdot E_t[\log_2 p_2(x_t \mid c_2(t)) - \log_2 p_{\text{blend}}(x_t \mid c_2(t))]"
        ),
        python_callable_module_path=(
            "submissions.hnerv_fec6_fixed_huffman_k16.encoder."
            "build_pr101_frame_exploit_selector_packet_fec10_hybrid:"
            "encode_fec10_hybrid_adaptive_blend"
        ),
        domain_of_validity={
            "n_pairs": {"min": 100, "max": 10000},
            "k_palette": {"min": 4, "max": 64},
            "alpha": {"min": 1, "max": 8},
            "h_marginal_bits_per_pair": {"min": 1.0, "max": 8.0},
            "transition_table_kind": ["static_hard_coded_zero_transmit"],
            "selector_index_stream_only": True,
            "in_domain_contexts": [
                "selector_index_stream_pr101_frame_exploit_600_pairs_k16_palette",
            ],
            "excluded_contexts": [
                "raw_payload_bytes_dense_streams",
                "neural_weight_tensors",
                "residual_correction_hybrid_substrates",
                "direct_byte_substitution_on_decode_opaque_raw_sections",
                # Per Catalog #359 sister discipline: adaptive-blend Markov is BEFORE-entropy-coder
                # context-model enhancement; NOT residual-correction-hybrid stacking-extension.
            ],
        },
        units_in={
            "n_pairs": "count",
            "k_palette": "count",
            "alpha": "dimensionless_smoothing_constant",
            "h_first_order_bits_per_pair": "bits/pair",
            "h_second_order_true_bits_per_pair": "bits/pair",
        },
        units_out={
            "fec10_hybrid_wire_bytes_total": "bytes",
            "delta_bytes_vs_fec6": "bytes",
            "delta_bytes_vs_fec8_2nd_order": "bytes",
            "delta_rate_contest_units": "contest_score_units",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "cascade_a_fec10_hybrid_alpha_2_live_fec6_stream_axis": 1.0,
        },
        last_calibration_utc=NOW_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
            "tools.cathedral_autopilot_autonomous_loop:rerank_candidates_via_master_gradient",
        ),
        canonical_producers=(
            "submissions.hnerv_fec6_fixed_huffman_k16.encoder."
            "build_pr101_frame_exploit_selector_packet_fec10_hybrid",
        ),
        provenance=eq_provenance,
    )


def main() -> None:
    eq_fec10 = _build_fec10_hybrid_equation()

    print(f"Registering {eq_fec10.equation_id!r}...")
    register_canonical_equation(eq_fec10)
    print(f"  ✓ Registered with {len(eq_fec10.empirical_anchors)} anchor(s)")

    print("\n=== Verification ===")
    from tac.canonical_equations import query_equations
    eqs = {e.equation_id for e in query_equations()}
    print(f"  {eq_fec10.equation_id}: {'PRESENT in registry' if eq_fec10.equation_id in eqs else 'MISSING'}")
    print(f"  Total registered equations: {len(eqs)}")


if __name__ == "__main__":
    main()
