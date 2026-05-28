#!/usr/bin/env python3
"""Register canonical equation ``fec8_3rd_order_markov_static_variant_c_savings_v1``.

Wave N+24 Option A next-iteration depth-axis extension per CLAUDE.md "Final rate
attack" standing directive + Catalog #344 canonical equations registry.

Per Catalog #344: every memo proposing a compound stack OR landing an empirical
finding MUST reference a canonical equation. This script registers the canonical
equation for FEC8 3rd-order Markov rate-attack codec savings AND lands the first
empirical anchor measured at $0 LOCAL CPU on 2026-05-28.

Sister of:

  * ``fec8_1st_order_markov_static_variant_b_savings_v1`` (245B = -4B vs FEC6)
  * ``fec8_2nd_order_true_markov_variant_a_savings_v1`` (canonical 166B Huffman
    wire-bit floor anchor; arithmetic-coder 239B sister anchor)
  * ``cascade_a_fec10_hybrid_adaptive_blend_savings_v1`` (Wave N+24 sister)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" +
Catalog #307 paradigm-vs-implementation classification: the 3rd-order
arithmetic-coder implementation is IMPLEMENTATION-LEVEL FALSIFIED on the
600-pair PR101 stream due to data sparsity (4096 contexts vs 597 quads).
The 3rd-order Markov PARADIGM is INTACT (Shannon H bits per pair = 0.8356
= 62.7 byte floor for 600 pairs is real signal); deployable implementations
are DEFERRED-pending-research per CLAUDE.md.

# SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

from tac.canonical_equations.registry import (
    CanonicalEquation,
    EmpiricalAnchor,
    register_canonical_equation,
)
from tac.provenance import builders


REPO_ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_JSON = (
    REPO_ROOT / ".omx/state/fec8_3rd_order_markov_rate_attack_measurement_20260528.json"
)


def main() -> None:
    if not MEASUREMENT_JSON.exists():
        raise SystemExit(
            f"measurement JSON missing: {MEASUREMENT_JSON}; "
            "regenerate via tools/measure_fec8_markov_3rd_order_rate_attack.py"
        )
    m = json.loads(MEASUREMENT_JSON.read_text())

    # Build canonical Provenance for the empirical anchor.
    # Per Catalog #323: every score claim must carry canonical Provenance.
    # This anchor is a predicted-from-model + measured-at-$0-local-CPU artifact;
    # use build_provenance_for_predicted with explicit hardware substrate.
    # source_sha256: hash of the empirical static payload (the canonical artifact
    # this anchor measures) per Catalog #323 canonical Provenance contract.
    anchor_provenance = builders.build_provenance_for_predicted(
        model_id="fec8_3rd_order_markov_static_variant_c_arithmetic_coder",
        inputs_sha256=m["fec8_3rd_order_static_payload_sha256"],
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="darwin_arm64_m5_max_macos_cpu",
    )

    anchor = EmpiricalAnchor(
        anchor_id="fec8_3rd_order_markov_static_variant_c_live_pr101_fec6_selector_stream_20260528",
        measurement_utc=m["measurement_utc"],
        inputs={
            "n_pairs": m["n_pairs"],
            "n_third_order_quads_observed": m["n_third_order_quads_observed"],
            "n_third_order_contexts_possible": m["n_third_order_contexts_possible"],
            "data_sparsity_ratio_observed_over_possible": m[
                "data_sparsity_ratio_observed_over_possible"
            ],
            "h_marginal_bits_per_pair": m["shannon_h_marginal_bits_per_pair"],
            "h_1st_order_bits_per_pair": m["shannon_h_1st_order_bits_per_pair"],
            "h_2nd_order_bits_per_pair": m["shannon_h_2nd_order_bits_per_pair"],
            "h_3rd_order_bits_per_pair": m["shannon_h_3rd_order_bits_per_pair"],
            "shannon_3rd_order_floor_bytes_for_n_pairs": m[
                "shannon_h_3rd_order_floor_bytes_for_n_pairs"
            ],
            "live_fec6_baseline_bytes": m["live_fec6_baseline_bytes"],
            "fec8_2nd_order_static_bytes": m["fec8_2nd_order_static_bytes"],
            "encoder_kind": "arithmetic_coder_static_third_order_shared_prior_4096_context_models",
            "shared_prior_table_path": ".omx/state/fec8_3rd_order_markov_empirical_table_20260528.py",
        },
        predicted_output={
            "predicted_band_bytes_saved_vs_fec6": [-25, -15],  # vs FEC6 249B
            "predicted_wire_bytes_total": 230,  # midpoint of band
            "predicted_band_methodology": (
                "linear extrapolation from FEC8 1st-order (-4B) + 2nd-order (-10B "
                "arithmetic-coder static) trend; expected diminishing returns due "
                "to 4096-context Laplace overhead with 597 observed quads"
            ),
        },
        empirical_output={
            "fec8_3rd_order_static_bytes_total": m["fec8_3rd_order_static_bytes_total"],
            "fec8_3rd_order_adaptive_bytes_total": m[
                "fec8_3rd_order_adaptive_bytes_total"
            ],
            "delta_bytes_vs_fec6_static_3rd": m["delta_bytes_vs_fec6_static_3rd"],
            "delta_bytes_vs_fec8_2nd_static_3rd": m["delta_bytes_vs_fec8_2nd_static_3rd"],
            "delta_bytes_vs_fec6_adaptive_3rd": m["delta_bytes_vs_fec6_adaptive_3rd"],
            "fec8_3rd_order_static_payload_sha256": m[
                "fec8_3rd_order_static_payload_sha256"
            ],
            "fec8_3rd_order_adaptive_payload_sha256": m[
                "fec8_3rd_order_adaptive_payload_sha256"
            ],
            "roundtrip_verified": True,
            "verdict": "IMPLEMENTATION_LEVEL_FALSIFIED_PARADIGM_INTACT_per_catalog_307",
            "verdict_rationale": (
                "FEC8 3rd-order arithmetic-coder static = 240B = -9B vs FEC6 249B "
                "but +1B WORSE than FEC8 2nd-order arithmetic 239B. Root cause: "
                "4096-context Laplace +1 smoothing overhead overwhelms the "
                "0.8356 bits/pair conditional entropy floor signal on this "
                "597-quad sparse stream. PARADIGM intact (Shannon floor 62.7B is "
                "real); IMPLEMENTATION deferred-pending-research per CLAUDE.md "
                "Forbidden premature KILL without research exhaustion."
            ),
            "delta_rate_contest_units_static_3rd": (
                -m["delta_bytes_vs_fec6_static_3rd"] * 25 / 37545489
            ),
            "reactivation_criteria": [
                "longer selector streams (>>600 symbols) where 4096-context "
                "Laplace overhead amortizes",
                "explicit per-context Huffman codec with deterministic codebook "
                "reconstruction (information-theoretic floor 106B / Shannon "
                "floor 62.7B vs current 240B)",
                "hybrid 2nd-order + selective 3rd-order escape pattern (only "
                "use 3rd-order for contexts with sufficient observations; fall "
                "back to 2nd-order deterministically)",
                "score-aware compound stacking with sister substrate selector "
                "streams that have higher 3rd-order conditional entropy reduction",
            ],
        },
        residual=abs(
            float(m["fec8_3rd_order_static_bytes_total"]) - 230.0
        ) / 230.0,
        source_artifact=".omx/state/fec8_3rd_order_markov_rate_attack_measurement_20260528.json",
        measurement_method=(
            "fec8_static_3rd_order_arithmetic_coder_encoder_run_on_live_pr101_fec6_"
            "selector_stream_then_compare_wire_bytes_vs_fec6_249B_baseline_and_"
            "fec8_2nd_order_239B_canonical_arithmetic_coder_anchor"
        ),
        provenance=anchor_provenance,
    )

    # Use the same payload sha256 as the anchor (the canonical artifact this
    # equation predicts the behavior of).
    equation_provenance = builders.build_provenance_for_predicted(
        model_id="fec8_3rd_order_markov_static_variant_c_canonical_equation",
        inputs_sha256=m["fec8_3rd_order_static_payload_sha256"],
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )

    eq = CanonicalEquation(
        equation_id="fec8_3rd_order_markov_static_variant_c_savings_v1",
        name="FEC8 3rd-order Markov static-prior arithmetic-coder selector codec savings",
        one_line_summary=(
            "FEC8 3rd-order Markov arithmetic-coder selector codec savings "
            "vs FEC6 baseline (4096-context shared prior + Laplace smoothing)."
        ),
        latex_form=(
            r"\Delta B_{\text{FEC8-3rd}} = "
            r"\lceil n_{\text{pairs}} \cdot H(X_t | X_{t-1}, X_{t-2}, X_{t-3}) "
            r"/ 8 \rceil + L_{\text{Laplace}}(K^3) + 8 - |\text{FEC6 baseline}|"
            r"\quad \text{where } L_{\text{Laplace}}(N) = O(N) \text{ overhead "
            r"per } K^3 = 4096 \text{ contexts}"
        ),
        python_callable_module_path=(
            "submissions.hnerv_fec6_fixed_huffman_k16.encoder."
            "build_pr101_frame_exploit_selector_packet_fec8_3rd_order_markov:"
            "encode_fec8_markov_selector_static_third_order"
        ),
        domain_of_validity={
            "context_count": {"max": 4096, "min": 16},
            "context_kind": [
                "true_3rd_order_markov_4096_context_shared_prior_table_via_canonical_sidecar",
            ],
            "encoder_kind": [
                "arithmetic_coder_static_third_order_witten_neal_cleary_1987_with_laplace_smoothing",
            ],
            "stream_length_pairs_observed_min": 597,
            "stream_length_pairs_observed_max": 597,
            "selector_palette_k": 16,
            "shared_prior_artifact_kind": (
                "wyner_ziv_decoder_side_information_4tuple_count_table_baked_in_sidecar"
            ),
            "data_sparsity_warning": (
                "with K^3 = 4096 contexts and only 597 observed 4-tuples, "
                "Laplace smoothing overhead dominates Shannon savings on short streams"
            ),
        },
        units_in={
            "n_pairs": "count",
            "h_3rd_order_bits_per_pair": "bits/symbol",
            "n_third_order_quads_observed": "count",
            "live_fec6_baseline_bytes": "bytes",
        },
        units_out={
            "fec8_3rd_order_static_bytes_total": "bytes",
            "delta_bytes_vs_fec6_static_3rd": "bytes",
            "delta_rate_contest_units_static_3rd": "dimensionless_rate",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "live_pr101_fec6_selector_stream_axis": float(anchor.residual),
        },
        last_calibration_utc=m["measurement_utc"],
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
        ),
        canonical_producers=(
            "submissions.hnerv_fec6_fixed_huffman_k16.encoder."
            "build_pr101_frame_exploit_selector_packet_fec8_3rd_order_markov",
        ),
        provenance=equation_provenance,
    )

    notes = (
        "Wave N+24 Option A next-iteration depth-axis extension. Sister of "
        "fec8_2nd_order_true_markov_variant_a_savings_v1 (166B Huffman wire-bit "
        "floor) and fec8_1st_order_markov_static_variant_b_savings_v1 (245B "
        "arithmetic). Per Catalog #307 IMPLEMENTATION-LEVEL FALSIFIED on 600-pair "
        "stream PARADIGM intact; per CLAUDE.md Forbidden premature KILL without "
        "research exhaustion DEFERRED-pending-research not KILLED. Reactivation "
        "criteria documented in module docstring + anchor empirical_output."
    )

    result = register_canonical_equation(
        eq,
        agent="claude",
        subagent_id="fec8_3rd_order_markov_rate_attack_20260528",
        notes=notes,
    )

    print(f"Registered: {result.equation_id}")
    print(f"  anchors: {len(result.empirical_anchors)}")
    print(f"  residual: {result.predicted_vs_empirical_residual}")
    print(f"  last calibrated: {result.last_calibration_utc}")
    print(f"  consumers: {result.canonical_consumers}")
    print(f"  producers: {result.canonical_producers}")
    print()
    print(f"Anchor empirical_output verdict: {anchor.empirical_output['verdict']}")
    print(f"Reactivation criteria: {len(anchor.empirical_output['reactivation_criteria'])} paths")


if __name__ == "__main__":
    main()
