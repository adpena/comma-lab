# SPDX-License-Identifier: MIT
"""One-shot canonical equation registration script (operator-approved 2026-05-26).

Per operator directive 2026-05-26 ("the modal and other operator decisions
from before and follow up are approved") + Catalog #344 operator-decision
protocol, registers 2 NEW canonical equations from today's empirical landings:

1. ``markov_context_selector_stream_compression_savings_v1`` — proposed by
   PR110-OPT-3 Variant B Markov context coder landing (commit 6474afde7;
   TaskCreate #1336). FEC8-static empirically achieved -4 bytes vs FEC6
   fixed-Huff (first method on this stream to beat fixed-Huff empirically).

2. ``residual_hybrid_boosting_savings_v1`` — proposed by BoostNeRV-PR110
   L1 EMPIRICAL respawn landing (commit b2fd3e587; TaskCreate #1337).
   gain_clamp=0.05 / 30ep produced 42-byte BPR1 sidecar / +0.0000280 Δrate;
   distortion-axis UNMEASURED. PROVISIONAL pending gain_clamp sweep
   disambiguator (TaskCreate #1342 in flight).

Both registrations carry canonical Provenance per Catalog #323 and
canonical_producers + canonical_consumers per Catalog #344 + CLAUDE.md
"Subagent coherence-by-default" producer→consumer audit. Per CLAUDE.md
"Pushing the frontier of research on optimization algorithms" standing
directive landed 2026-05-26: these registrations are first-class
research-contribution deliverables.
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


def _build_markov_context_equation() -> CanonicalEquation:
    """FEC8 1st-order Markov context coder on K=16 selector index stream."""
    anchor_provenance = build_provenance_for_predicted(
        model_id="fec8_markov_context_coder_v1_static_transition_table",
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_m5_max_macos_cpu_advisory",
        captured_at_utc="2026-05-26T17:55:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="markov_context_fec8_static_live_fec6_stream_20260526",
        measurement_utc="2026-05-26T17:55:00Z",
        inputs={
            "n_pairs": 600,
            "k_palette": 16,
            "k_contexts": 16,
            "h_marginal_bits_per_pair": 3.2116,
            "h_conditional_bits_per_pair": 2.9402,
            "transition_table_kind": "static_hard_coded_zero_transmit",
            "live_stream_archive_sha256": (
                "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
            ),
        },
        predicted_output={"savings_bytes_wire": 10.25},
        empirical_output={"savings_bytes_wire": 4.0},
        residual=6.25,  # normalized magnitude per CanonicalEquation contract (signed-direction: empirical < predicted → equation overestimates savings on this stream)
        source_artifact=(
            ".omx/research/pr110_opt3_variant_b_markov_landed_20260526.md"
        ),
        measurement_method=(
            "fec8_static_encoder_run_on_live_fec6_selector_stream_then_compare_wire_bytes"
        ),
        provenance=anchor_provenance,
    )
    eq_provenance = build_provenance_for_predicted(
        model_id="markov_context_selector_stream_compression_savings_v1",
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic",
        captured_at_utc=NOW_UTC,
    )
    return CanonicalEquation(
        equation_id="markov_context_selector_stream_compression_savings_v1",
        name="Markov-context selector-stream compression savings (FEC8)",
        one_line_summary=(
            "Predicts wire-byte savings vs fixed-Huffman when encoding K-symbol "
            "selector index stream with 1st-order Markov context arithmetic coder "
            "(static hard-coded transition table; zero transmit)"
        ),
        latex_form=(
            r"\Delta B = \frac{N}{8}(H(X) - H(X_t \mid X_{t-1})) - \text{header\_overhead} "
            r"- \text{cacm87\_termination} - \text{byte\_padding\_lsb\_loss} "
            r"- \text{alphabet\_fragmentation\_loss}(N, K_{contexts})"
        ),
        python_callable_module_path=(
            "submissions.hnerv_fec6_fixed_huffman_k16.encoder."
            "build_pr101_frame_exploit_selector_packet_markov:"
            "predict_fec8_static_wire_savings"
        ),
        domain_of_validity={
            "n_pairs": {"min": 100, "max": 10000},
            "k_palette": {"min": 4, "max": 64},
            "h_marginal_bits_per_pair": {"min": 1.0, "max": 8.0},
            "transition_table_kind": ["static_hard_coded_zero_transmit", "adaptive_online"],
            "selector_index_stream_only": True,
            "excluded_contexts": [
                "raw_payload_bytes_dense_streams",
                "neural_weight_tensors",
                "residual_correction_hybrid_substrates",
            ],
        },
        units_in={
            "n_pairs": "count",
            "k_palette": "count",
            "k_contexts": "count",
            "h_marginal_bits_per_pair": "bits/pair",
            "h_conditional_bits_per_pair": "bits/pair",
        },
        units_out={"savings_bytes_wire": "bytes"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"fec8_static_live_fec6_stream_axis": 6.25},  # normalized magnitude per CanonicalEquation contract
        last_calibration_utc=NOW_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
        ),
        canonical_producers=(
            "submissions.hnerv_fec6_fixed_huffman_k16.encoder."
            "build_pr101_frame_exploit_selector_packet_markov",
        ),
        provenance=eq_provenance,
    )


def _build_residual_hybrid_boosting_equation() -> CanonicalEquation:
    """BoostNeRV residual-correction hybrid sidecar bytes (PROVISIONAL)."""
    anchor_provenance = build_provenance_for_predicted(
        model_id="boostnerv_pr110_residual_correction_hybrid_gain_clamp_0p05_30ep",
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="darwin_arm64_m5_max_mlx_local",
        captured_at_utc="2026-05-26T18:20:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="residual_hybrid_boosting_boostnerv_pr110_clamp_0p05_30ep_20260526",
        measurement_utc="2026-05-26T18:20:00Z",
        inputs={
            "substrate": "boost_nerv_pr110",
            "gain_clamp": 0.05,
            "epochs": 30,
            "pairs": 50,
            "spatial_size_hw": [96, 128],
            "brotli_quality": 9,
            "base_archive_sha256_pr110_fec6": (
                "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
            ),
            "base_substrate_size_bytes_int8": 1843200,
        },
        predicted_output={
            "bpr1_sidecar_bytes_total": 42,
            "delta_rate_contest_units": 2.80e-5,
        },
        empirical_output={
            "bpr1_sidecar_bytes_total": 42,
            "delta_rate_contest_units": 2.80e-5,
            "distortion_axis_measurement_status": "UNMEASURED_pending_paired_cuda_or_mlx_segnet_posenet_routing",
        },
        residual=0.0,
        source_artifact=".omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md",
        measurement_method=(
            "boostnerv_pr110_mlx_local_30ep_50pairs_96x128_then_brotli_quality_9_sidecar"
        ),
        provenance=anchor_provenance,
    )
    eq_provenance = build_provenance_for_predicted(
        model_id="residual_hybrid_boosting_savings_v1",
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_provisional",
        captured_at_utc=NOW_UTC,
    )
    return CanonicalEquation(
        equation_id="residual_hybrid_boosting_savings_v1",
        name="Residual-correction hybrid boosting savings (BoostNeRV-class)",
        one_line_summary=(
            "Predicts BPR1 sidecar byte cost + Δrate when residual-correction hybrid encoder "
            "applied to substrate per-pair residual; PROVISIONAL pending distortion-axis paired measurement"
        ),
        latex_form=(
            r"B_{sidecar} = \text{brotli}_q(\text{int8\_residual\_clamped}(\theta_{substrate}, "
            r"\text{gain\_clamp})) + B_{header}; \Delta_{rate} = 25 B_{sidecar} / 37545489"
        ),
        python_callable_module_path=(
            "src.tac.substrates.boost_nerv.residual_correction:"
            "predict_residual_hybrid_sidecar_bytes_and_delta_rate"
        ),
        domain_of_validity={
            "substrate_class": [
                "boost_nerv_pr110",
                "nirvana_cascading_nerv",
                "cascading_residual_decoder_class",
            ],
            "gain_clamp": {"min": 0.01, "max": 1.0},
            "epochs": {"min": 30, "max": 1000},
            "brotli_quality": {"min": 6, "max": 11},
            "base_substrate_size_bytes": {"min": 100000, "max": 100000000},
            "distortion_axis_unmeasured_provisional": True,
            "excluded_contexts": [
                "selector_index_streams",
                "chroma_lut_replacement",
                "direct_lookup_substitution",
            ],
        },
        units_in={
            "gain_clamp": "dimensionless",
            "epochs": "count",
            "pairs": "count",
            "base_substrate_size_bytes_int8": "bytes",
            "brotli_quality": "level",
        },
        units_out={
            "bpr1_sidecar_bytes_total": "bytes",
            "delta_rate_contest_units": "contest_score_units",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "boostnerv_pr110_clamp_0p05_30ep_axis": 0.0,
            "distortion_axis_unmeasured": float("nan"),
        },
        last_calibration_utc=NOW_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
        ),
        canonical_producers=(
            "src.tac.substrates.boost_nerv.residual_correction",
            "experiments.train_substrate_boost_nerv",
        ),
        provenance=eq_provenance,
    )


def main() -> None:
    eq_markov = _build_markov_context_equation()
    eq_residual = _build_residual_hybrid_boosting_equation()

    print(f"Registering {eq_markov.equation_id!r}...")
    register_canonical_equation(eq_markov)
    print(f"  ✓ Registered with {len(eq_markov.empirical_anchors)} anchor(s)")

    print(f"Registering {eq_residual.equation_id!r}...")
    register_canonical_equation(eq_residual)
    print(f"  ✓ Registered with {len(eq_residual.empirical_anchors)} anchor(s) (PROVISIONAL)")

    print("\n=== Verification ===")
    from tac.canonical_equations import query_equations
    eqs = {e.equation_id for e in query_equations()}
    for eid in (eq_markov.equation_id, eq_residual.equation_id):
        print(f"  {eid}: {'PRESENT in registry' if eid in eqs else 'MISSING'}")


if __name__ == "__main__":
    main()
